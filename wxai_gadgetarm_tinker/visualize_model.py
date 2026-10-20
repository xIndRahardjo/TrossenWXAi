import os
import sys
import time
import argparse
import numpy as np
import torch
import mujoco

sys.path.insert(0, os.path.dirname(__file__))
from wxai_gadgetarm_env import WxaiGadgetArmEnv
from wxai_gadgetarm_env_stage2_gripping import WxaiGadgetArmStage2Env
from ppo_agent import PPOAgent

def run_visualization(stage="combined", stage1_model=None, stage2_model=None, num_episodes=5, real_time=True):
    exp_dir = os.path.dirname(__file__)
    
    default_stage1_model = os.path.join(exp_dir, "best_gadgetarm_model.pt")
    if not os.path.exists(default_stage1_model):
        default_stage1_model = os.path.join(exp_dir, "final_gadgetarm_model.pt")

    default_stage2_model = os.path.join(exp_dir, "best_stage2_gripping_model.pt")
    if not os.path.exists(default_stage2_model):
        default_stage2_model = os.path.join(exp_dir, "final_stage2_gripping_model.pt")

    if stage1_model is None:
        stage1_model = default_stage1_model
    if stage2_model is None:
        stage2_model = default_stage2_model

    stage_str = str(stage).lower()
    
    if stage_str in ["1", "stage1", "approach"]:
        env = WxaiGadgetArmEnv()
        print(f"==================================================")
        print(f" Testing Stage 1 RL Model (Approach & Reaching)")
        print(f" Model Checkpoint: {stage1_model}")
        print(f"==================================================")
        agent1 = PPOAgent(env.observation_dim, env.action_dim, device="cpu")
        if os.path.exists(stage1_model):
            agent1.load(stage1_model)
            print("Loaded Stage 1 checkpoint successfully!")
        agent2 = None
    elif stage_str in ["2", "stage2", "gripping"]:
        env = WxaiGadgetArmStage2Env()
        print(f"==================================================")
        print(f" Testing Stage 2 RL Model (Gripping Focus)")
        print(f" Model Checkpoint: {stage2_model}")
        print(f"==================================================")
        agent1 = None
        agent2 = PPOAgent(env.observation_dim, env.action_dim, device="cpu")
        if os.path.exists(stage2_model):
            agent2.load(stage2_model)
            print("Loaded Stage 2 checkpoint successfully!")
    else: # Combined / full pipeline
        env = WxaiGadgetArmEnv()
        print(f"==================================================")
        print(f" Testing Combined Pipeline (Stage 1 Approach -> Stage 2 Gripping)")
        print(f" Stage 1 Model: {stage1_model}")
        print(f" Stage 2 Model: {stage2_model}")
        print(f"==================================================")
        agent1 = PPOAgent(env.observation_dim, env.action_dim, device="cpu")
        if os.path.exists(stage1_model):
            agent1.load(stage1_model)
            print("Loaded Stage 1 checkpoint successfully!")
            
        agent2 = PPOAgent(env.observation_dim, env.action_dim, device="cpu")
        if os.path.exists(stage2_model):
            agent2.load(stage2_model)
            print("Loaded Stage 2 checkpoint successfully!")

    try:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for ep in range(num_episodes):
                obs, _ = env.reset()
                print(f"\n--- Episode {ep+1}/{num_episodes} ---")
                done = False
                step = 0
                ep_reward = 0.0
                info = {}
                current_mode = "APPROACH (Stage 1)" if agent1 is not None else "GRIPPING (Stage 2)"
                switched_to_stage2 = False
                stage2_steps = 0
                max_stage2_steps = 80

                while not done and viewer.is_running():
                    step_start = time.time()

                    if current_mode.startswith("APPROACH") and agent1 is not None:
                        action, _, _ = agent1.select_action(obs, deterministic=True)
                    else:
                        action, _, _ = agent2.select_action(obs, deterministic=True)

                    obs, reward, done, _, info = env.step(action)
                    ep_reward += reward
                    step += 1

                    # Transition from Stage 1 Approach to Stage 2 Gripping upon reaching pre-grasp goal
                    if agent1 is not None and agent2 is not None and not switched_to_stage2:
                        if info.get("object_inside_gripper", False) or info.get("success", False):
                            switched_to_stage2 = True
                            current_mode = "GRIPPING (Stage 2)"
                            done = False
                            print(f" Step {step:3d}: tinker.obj centered inside open gripper (d_xy={info.get('d_xy', 0):.4f}m, h={info.get('ee_height', 0):.4f}m). Switching to Stage 2 Gripping Policy!")

                    if switched_to_stage2:
                        stage2_steps += 1
                        left_touch, right_touch = env.check_finger_contacts()
                        if left_touch and right_touch:
                            info["stage2_grip_success"] = True
                        if stage2_steps >= max_stage2_steps or info.get("dropped", False):
                            done = True

                    viewer.sync()

                    if real_time:
                        time_until_next_frame = env.dt - (time.time() - step_start)
                        if time_until_next_frame > 0:
                            time.sleep(time_until_next_frame)

                left_touch, right_touch = env.check_finger_contacts()
                print(f"Episode {ep+1} Finished | Total Steps: {step} | Total Reward: {ep_reward:.2f}")
                print(f"Final Status | Mode: {current_mode} | Gripper Contacts: Left={left_touch}, Right={right_touch} | Info: {info}")
    except Exception as e:
        print(f"Note: Interactive GUI viewer skipped ({e}). Running simulation loop...")
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            step = 0
            ep_reward = 0.0
            info = {}
            current_mode = "APPROACH (Stage 1)" if agent1 is not None else "GRIPPING (Stage 2)"
            switched_to_stage2 = False
            stage2_steps = 0
            max_stage2_steps = 80

            while not done:
                if current_mode.startswith("APPROACH") and agent1 is not None:
                    action, _, _ = agent1.select_action(obs, deterministic=True)
                else:
                    action, _, _ = agent2.select_action(obs, deterministic=True)

                obs, reward, done, _, info = env.step(action)
                ep_reward += reward
                step += 1

                # Transition from Stage 1 Approach to Stage 2 Gripping upon reaching pre-grasp goal
                if agent1 is not None and agent2 is not None and not switched_to_stage2:
                    if info.get("object_inside_gripper", False) or info.get("success", False):
                        switched_to_stage2 = True
                        current_mode = "GRIPPING (Stage 2)"
                        done = False
                        print(f" Step {step:3d}: tinker.obj centered inside open gripper (d_xy={info.get('d_xy', 0):.4f}m, h={info.get('ee_height', 0):.4f}m). Switching to Stage 2 Gripping Policy!")

                if switched_to_stage2:
                    stage2_steps += 1
                    left_touch, right_touch = env.check_finger_contacts()
                    if left_touch and right_touch:
                        info["stage2_grip_success"] = True
                    if stage2_steps >= max_stage2_steps or info.get("dropped", False):
                        done = True

            left_touch, right_touch = env.check_finger_contacts()
            print(f"Episode {ep+1} Finished | Total Steps: {step} | Total Reward: {ep_reward:.2f} | Info: {info}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RL models in MuJoCo simulation viewer (Stage 1, Stage 2, or Combined)")
    parser.add_argument("--stage", type=str, default="combined", choices=["1", "2", "combined"], help="Pipeline stage to test: '1' (Approach), '2' (Gripping), or 'combined' (Full pipeline)")
    parser.add_argument("--stage1_model", type=str, default=None, help="Path to Stage 1 .pt model checkpoint")
    parser.add_argument("--stage2_model", type=str, default=None, help="Path to Stage 2 .pt model checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--no-real-time", action="store_true", help="Run at max simulation speed without real-time delay")
    
    args = parser.parse_args()
    run_visualization(
        stage=args.stage,
        stage1_model=args.stage1_model,
        stage2_model=args.stage2_model,
        num_episodes=args.episodes,
        real_time=not args.no_real_time
    )
