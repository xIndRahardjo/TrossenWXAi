import os
import sys
import time
import json
import argparse
import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))
from wxai_gadgetarm_env import WxaiGadgetArmEnv
from ppo_agent import PPOAgent

def train_gadgetarm(total_steps=500000, buffer_size=12000, batch_size=800, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- GadgetArm Experiment Training (WXAI Arm & tinker.obj) ---")
    print(f"Device: {device}")
    print(f"Total Steps: {total_steps}, Buffer Size: {buffer_size}, Batch Size: {batch_size}")

    env = WxaiGadgetArmEnv()
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
    best_model_path = os.path.join(output_dir, "best_gadgetarm_model.pt")
    final_model_path = os.path.join(output_dir, "final_gadgetarm_model.pt")
    log_json_path = os.path.join(output_dir, "training_log.json")

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
        ep_lifts = []
        curr_ep_reward = 0.0
        curr_ep_len = 0

        # Collect 12,000 steps buffer (as per paper Table 1)
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
                ep_lifts.append(float(info.get("is_lifted", False)))
                curr_ep_reward = 0.0
                curr_ep_len = 0
                obs, _ = env.reset()
            else:
                obs = next_obs

            if total_timesteps >= total_steps:
                break

        # Compute next value for GAE
        _, _, next_value = agent.select_action(obs)
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

        # Update PPO policy
        agent.update(states, actions, log_probs, returns, advantages)

        avg_reward = np.mean(ep_rewards) if len(ep_rewards) > 0 else 0.0
        avg_len = np.mean(ep_lengths) if len(ep_lengths) > 0 else 0.0
        success_rate = np.mean(ep_successes) if len(ep_successes) > 0 else 0.0
        lift_rate = np.mean(ep_lifts) if len(ep_lifts) > 0 else 0.0

        print(
            f"Iter {iteration:03d} | Timesteps: {total_timesteps}/{total_steps} | "
            f"Avg Reward: {avg_reward:.2f} | Avg Len: {avg_len:.1f} | "
            f"Grasp/Lift Rate: {lift_rate*100:.1f}% | Success Rate: {success_rate*100:.1f}%"
        )

        log_entry = {
            "iteration": iteration,
            "timesteps": total_timesteps,
            "avg_reward": float(avg_reward),
            "avg_length": float(avg_len),
            "lift_rate": float(lift_rate),
            "success_rate": float(success_rate)
        }
        training_logs.append(log_entry)

        best_score = getattr(agent, "best_score", -float("inf"))
        current_score = success_rate * 10000.0 + avg_reward
        if current_score > best_score:
            agent.best_score = current_score
            best_reward = avg_reward
            agent.save(best_model_path)
            print(f"  --> Saved new best Stage 1 model checkpoint to {best_model_path} (Success Rate: {success_rate*100:.1f}%, Avg Reward: {avg_reward:.2f})")

        with open(log_json_path, "w") as f:
            json.dump(training_logs, f, indent=2)

    agent.save(final_model_path)
    print(f"\nTraining completed! Saved final model to {final_model_path}")
    print(f"Logged training stats to {log_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GadgetArm WXAI arm PPO agent on tinker.obj")
    parser.add_argument("--total_steps", type=int, default=500000, help="Total timesteps to train (default: 500000)")
    parser.add_argument("--buffer_size", type=int, default=12000, help="Buffer size per PPO iteration (default: 12000)")
    parser.add_argument("--batch_size", type=int, default=800, help="Batch size for PPO update (default: 800)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    train_gadgetarm(
        total_steps=args.total_steps,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        seed=args.seed
    )
