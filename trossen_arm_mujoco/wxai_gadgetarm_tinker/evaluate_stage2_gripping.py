import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from wxai_gadgetarm_env_stage2_gripping import WxaiGadgetArmStage2Env
from ppo_agent import PPOAgent

def evaluate_and_plot_stage2(num_episodes=20, render_video=False):
    exp_dir = os.path.dirname(__file__)
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    log_path = os.path.join(exp_dir, "training_log_stage2.json")
    model_path = os.path.join(exp_dir, "best_stage2_gripping_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(exp_dir, "final_stage2_gripping_model.pt")

    print(f"--- GadgetArm Experiment Stage 2 Evaluation (Gripping Focus) ---")
    print(f"Loading checkpoint: {model_path}")
    print(f"Loading training logs: {log_path}")

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            logs = json.load(f)

        steps = [entry["timesteps"] for entry in logs]
        rewards = [entry["avg_reward"] for entry in logs]
        success_rates = [entry["success_rate"] * 100.0 for entry in logs]

        # Plot 1: Stage 2 Reward Curve
        plt.figure(figsize=(9, 5))
        plt.plot(steps, rewards, label="Stage 2 PPO Cumulative Reward", color="#9467bd", linewidth=2.0)
        plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
        plt.title("GadgetArm Stage 2: PPO Reward Curve (Gripping Focus)", fontsize=12, fontweight="bold")
        plt.xlabel("Training Steps", fontsize=11)
        plt.ylabel("Average Episode Reward", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=10)
        reward_plot_path = os.path.join(plots_dir, "stage2_gripping_rewards.png")
        plt.savefig(reward_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Stage 2 reward plot to {reward_plot_path}")

        # Plot 2: Dual Contact Success Rate
        plt.figure(figsize=(9, 5))
        plt.plot(steps, success_rates, label="Dual-Finger Contact Success (%)", color="#2ca02c", linewidth=2.0)
        plt.title("GadgetArm Stage 2: Stable Gripping Success Rate vs Steps", fontsize=12, fontweight="bold")
        plt.xlabel("Training Steps", fontsize=11)
        plt.ylabel("Success Percentage (%)", fontsize=11)
        plt.ylim(-5, 105)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=10)
        success_plot_path = os.path.join(plots_dir, "stage2_contact_success.png")
        plt.savefig(success_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Stage 2 success plot to {success_plot_path}")

    # Evaluate Policy Rollouts
    env = WxaiGadgetArmStage2Env()
    agent = PPOAgent(env.observation_dim, env.action_dim, device="cpu")
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print("Warning: No Stage 2 model checkpoint found! Evaluating initial policy.")

    total_rewards = []
    successes = []
    touch_durations = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            ep_reward += reward

        total_rewards.append(ep_reward)
        successes.append(info.get("success", False))
        touch_durations.append(info.get("consecutive_touch_count", 0))

    print(f"\n--- Stage 2 Evaluation Results ({num_episodes} Episodes) ---")
    print(f"Mean Episode Reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"Mean Consecutive Contact Duration: {np.mean(touch_durations):.1f} steps")
    print(f"Stable Dual-Finger Gripping Success Rate: {np.mean(successes)*100.0:.1f}%")

    if render_video:
        try:
            import mujoco
            renderer = mujoco.Renderer(env.model, height=480, width=640)
            obs, _ = env.reset()
            frames = []

            for step in range(80):
                action, _, _ = agent.select_action(obs, deterministic=True)
                obs, _, done, _, _ = env.step(action)
                renderer.update_scene(env.data)
                pixels = renderer.render()
                frames.append(pixels)
                if done:
                    break

            if len(frames) > 0:
                import imageio
                gif_path = os.path.join(plots_dir, "stage2_gripping_demo.gif")
                imageio.mimsave(gif_path, frames, fps=20)
                print(f"Saved Stage 2 demo animation to {gif_path}")
        except Exception as e:
            print(f"Note: Visual rendering skipped or optional package missing: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 (Gripping) model")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes (default: 20)")
    parser.add_argument("--video", action="store_true", help="Render Stage 2 demo GIF")
    args = parser.parse_args()

    evaluate_and_plot_stage2(num_episodes=args.episodes, render_video=args.video)
