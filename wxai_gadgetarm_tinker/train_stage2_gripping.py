import os
import sys
import time
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from wxai_gadgetarm_env_stage2_gripping import WxaiGadgetArmStage2Env
from ppo_agent import PPOAgent

def train_stage2_gripping(total_steps=200000, buffer_size=8000, batch_size=800, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- GadgetArm Experiment Stage 2 Training (Gripping Focus) ---")
    print(f"Device: {device}")
    print(f"Total Steps: {total_steps}, Buffer Size: {buffer_size}, Batch Size: {batch_size}")

    env = WxaiGadgetArmStage2Env()
    state_dim = env.observation_dim
    action_dim = env.action_dim

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3.0e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        beta_entropy=0.005,
        batch_size=batch_size,
        device=device
    )

    output_dir = os.path.dirname(__file__)
    best_model_path = os.path.join(output_dir, "best_stage2_gripping_model.pt")
    final_model_path = os.path.join(output_dir, "final_stage2_gripping_model.pt")
    log_json_path = os.path.join(output_dir, "training_log_stage2.json")

    total_timesteps = 0
    iteration = 0
    best_reward = -float("inf")

    training_logs = []

    obs, _ = env.reset()

    while total_timesteps < total_steps:
        iteration += 1
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        ep_rewards = []
        ep_lengths = []
        ep_successes = []
        ep_touch_counts = []
        ep_lifts = []
        curr_ep_reward = 0.0
        curr_ep_len = 0

        for _ in range(buffer_size):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, _, info = env.step(action)

            states.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(float(done))

            curr_ep_reward += reward
            curr_ep_len += 1
            total_timesteps += 1

            if done:
                ep_rewards.append(curr_ep_reward)
                ep_lengths.append(curr_ep_len)
                ep_successes.append(float(info.get("success", False)))
                ep_touch_counts.append(info.get("consecutive_touch_count", 0))
                ep_lifts.append(float(info.get("is_lifted", False)))
                curr_ep_reward = 0.0
                curr_ep_len = 0
                obs, _ = env.reset()
            else:
                obs = next_obs

            if total_timesteps >= total_steps:
                break

        _, _, next_value = agent.select_action(obs)
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

        agent.update(states, actions, log_probs, returns, advantages)

        avg_reward = np.mean(ep_rewards) if len(ep_rewards) > 0 else 0.0
        avg_len = np.mean(ep_lengths) if len(ep_lengths) > 0 else 0.0
        success_rate = np.mean(ep_successes) if len(ep_successes) > 0 else 0.0
        lift_rate = np.mean(ep_lifts) if len(ep_lifts) > 0 else 0.0
        avg_touch_count = np.mean(ep_touch_counts) if len(ep_touch_counts) > 0 else 0.0

        print(
            f"Iter {iteration:03d} | Timesteps: {total_timesteps}/{total_steps} | "
            f"Avg Reward: {avg_reward:.2f} | Lift Rate: {lift_rate*100:.1f}% | "
            f"Avg Touch Count: {avg_touch_count:.1f} steps | Gripping & Uplift Success: {success_rate*100:.1f}%"
        )

        log_entry = {
            "iteration": iteration,
            "timesteps": total_timesteps,
            "avg_reward": float(avg_reward),
            "avg_length": float(avg_len),
            "avg_touch_count": float(avg_touch_count),
            "lift_rate": float(lift_rate),
            "success_rate": float(success_rate)
        }
        training_logs.append(log_entry)

        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(best_model_path)
            print(f"  --> Saved new best Stage 2 model checkpoint to {best_model_path} (Reward: {best_reward:.2f})")

        with open(log_json_path, "w") as f:
            json.dump(training_logs, f, indent=2)

    agent.save(final_model_path)
    print(f"\nStage 2 Training completed! Saved final model to {final_model_path}")
    print(f"Logged training stats to {log_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 2 (Gripping) PPO Agent")
    parser.add_argument("--total_steps", type=int, default=200000, help="Total timesteps (default: 200000)")
    parser.add_argument("--buffer_size", type=int, default=8000, help="Buffer size (default: 8000)")
    parser.add_argument("--batch_size", type=int, default=800, help="Batch size (default: 800)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    train_stage2_gripping(
        total_steps=args.total_steps,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        seed=args.seed
    )
