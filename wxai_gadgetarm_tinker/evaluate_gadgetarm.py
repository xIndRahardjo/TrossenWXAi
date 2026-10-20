import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from wxai_gadgetarm_env import WxaiGadgetArmEnv
from ppo_agent import PPOAgent

def evaluate_and_plot(num_episodes=20, render_video=False):
    exp_dir = os.path.dirname(__file__)
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    log_path = os.path.join(exp_dir, "training_log.json")
    model_path = os.path.join(exp_dir, "best_gadgetarm_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(exp_dir, "final_gadgetarm_model.pt")

    print(f"--- GadgetArm Experiment Evaluation ---")
    print(f"Loading checkpoint: {model_path}")
    print(f"Loading training logs: {log_path}")

    # 1. Plot Training Curves from training_log.json if available
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            logs = json.load(f)

        steps = [entry["timesteps"] for entry in logs]
        rewards = [entry["avg_reward"] for entry in logs]
        lift_rates = [entry["lift_rate"] * 100.0 for entry in logs]
        success_rates = [entry["success_rate"] * 100.0 for entry in logs]

        # Plot 1: Cumulative Reward Curve
        plt.figure(figsize=(9, 5))
        plt.plot(steps, rewards, label="PPO Cumulative Reward", color="#1f77b4", linewidth=2.0)
        plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
        plt.title("GadgetArm Experiment: PPO Cumulative Reward (WXAI Arm & tinker.obj)", fontsize=12, fontweight="bold")
        plt.xlabel("Training Steps", fontsize=11)
        plt.ylabel("Average Episode Reward", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=10)
        reward_plot_path = os.path.join(plots_dir, "gadgetarm_training_rewards.png")
        plt.savefig(reward_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved reward plot to {reward_plot_path}")

        # Plot 2: Grasp & Lift Success Rate
        plt.figure(figsize=(9, 5))
        plt.plot(steps, lift_rates, label="Grasp & Lift Rate (%)", color="#2ca02c", linewidth=2.0)
        plt.plot(steps, success_rates, label="Full Task Success Rate (%)", color="#ff7f0e", linewidth=2.0, linestyle="--")
        plt.title("GadgetArm Experiment: Grasp & Task Success Rate vs Steps", fontsize=12, fontweight="bold")
        plt.xlabel("Training Steps", fontsize=11)
        plt.ylabel("Success Percentage (%)", fontsize=11)
        plt.ylim(-5, 105)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=10)
        success_plot_path = os.path.join(plots_dir, "gadgetarm_grasp_success.png")
        plt.savefig(success_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved success plot to {success_plot_path}")

    # 2. Evaluate Policy Rollouts
    env = WxaiGadgetArmEnv()
    agent = PPOAgent(env.observation_dim, env.action_dim, device="cpu")
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print("Warning: No model checkpoint found! Evaluating un-trained / random policy.")

    total_rewards = []
    successes = []
    lifts = []

    trajectory_history = {
        "ee_z": [],
        "obj_z": [],
        "d_ee_obj": [],
        "d_obj_target": []
    }

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        step = 0

        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            ep_reward += reward
            step += 1

            if ep == 0:
                ee_pos = env.get_ee_position()
                obj_pos = env.get_object_position()
                trajectory_history["ee_z"].append(ee_pos[2])
                trajectory_history["obj_z"].append(obj_pos[2])
                trajectory_history["d_ee_obj"].append(info["d_eef_obj"])
                trajectory_history["d_obj_target"].append(info["d_obj_target"])

        total_rewards.append(ep_reward)
        successes.append(info.get("success", False) or info.get("object_inside_gripper", False))
        lifts.append(info.get("is_lifted", False))

    print(f"\n--- Stage 1 Evaluation Results ({num_episodes} Episodes) ---")
    print(f"Mean Episode Reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"Stage 1 Reach & Alignment Success Rate: {np.mean(successes)*100.0:.1f}%")

    # Plot 3: Detailed Trajectory Profile (Episode 1)
    if len(trajectory_history["ee_z"]) > 0:
        plt.figure(figsize=(10, 6))
        steps_axis = np.arange(len(trajectory_history["ee_z"])) * env.dt
        plt.plot(steps_axis, trajectory_history["ee_z"], label="End-Effector Z (m)", color="blue", linewidth=2.0)
        plt.plot(steps_axis, trajectory_history["obj_z"], label="tinker.obj Z (m)", color="orange", linewidth=2.0)
        plt.plot(steps_axis, trajectory_history["d_ee_obj"], label="EE-to-Object Dist (m)", color="green", linestyle="--")
        plt.axhline(env.table_z, color="gray", linestyle=":", label="Table Surface Height")

        plt.title("GadgetArm Trajectory Elevation Profile (WXAI Arm & tinker.obj)", fontsize=12, fontweight="bold")
        plt.xlabel("Time (seconds)", fontsize=11)
        plt.ylabel("Height / Distance (meters)", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=10)
        traj_plot_path = os.path.join(plots_dir, "gadgetarm_trajectory.png")
        plt.savefig(traj_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved trajectory profile plot to {traj_plot_path}")

    # 3. Optional Video / Image Rendering
    if render_video:
        try:
            import cv2
            renderer = cv2
            video_path = os.path.join(plots_dir, "gadgetarm_grasp_demo.mp4")
            obs, _ = env.reset()
            frames = []
            
            # Using MuJoCo offscreen rendering
            import mujoco
            renderer = mujoco.Renderer(env.model, height=480, width=640)

            for step in range(120):
                action, _, _ = agent.select_action(obs, deterministic=True)
                obs, _, done, _, _ = env.step(action)
                renderer.update_scene(env.data)
                pixels = renderer.render()
                frames.append(pixels)
                if done:
                    break

            if len(frames) > 0:
                import imageio
                imageio.mimsave(os.path.join(plots_dir, "gadgetarm_grasp_demo.gif"), frames, fps=25)
                print(f"Saved demo animation to {os.path.join(plots_dir, 'gadgetarm_grasp_demo.gif')}")
        except Exception as e:
            print(f"Note: Visual rendering skipped or optional package missing: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GadgetArm experiment model")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes (default: 20)")
    parser.add_argument("--video", action="store_true", help="Render demo GIF/video")
    args = parser.parse_args()

    evaluate_and_plot(num_episodes=args.episodes, render_video=args.video)
