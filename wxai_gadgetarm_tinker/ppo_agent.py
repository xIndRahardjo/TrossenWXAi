import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Policy Network (Actor)
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # Value Network (Critic)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        mean = self.actor_mean(state)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        value = self.critic(state)
        return mean, std, value

    def get_action(self, state, deterministic=False):
        mean, std, value = self.forward(state)
        if deterministic:
            return mean, torch.zeros_like(mean), value
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, value

    def evaluate_actions(self, state, action):
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy, value


class PPOAgent:
    """
    PPO Agent implementation matching the hyperparameters specified in Table 1
    of sensors-20-06183-v2.pdf:
    - Learning rate: 3.0e-4
    - Batch Size: 800
    - Buffer Size: 12,000
    - Beta (Entropy Coeff): 0.005
    - Epsilon (Clip ratio): 0.2
    - Lambda (GAE): 0.95
    - Gamma (Discount factor): 0.99
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3.0e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        beta_entropy=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=10,
        batch_size=800,
        device="cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.beta_entropy = beta_entropy
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    def select_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_t, log_prob_t, value_t = self.policy.get_action(state_t, deterministic)
        return (
            action_t.cpu().numpy().flatten(),
            log_prob_t.cpu().numpy().flatten()[0],
            value_t.cpu().numpy().flatten()[0]
        )

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0.0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)

    def update(self, states, actions, old_log_probs, returns, advantages):
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs_t = torch.FloatTensor(np.array(old_log_probs)).unsqueeze(-1).to(self.device)
        returns_t = torch.FloatTensor(np.array(returns)).unsqueeze(-1).to(self.device)
        advantages_t = torch.FloatTensor(np.array(advantages)).unsqueeze(-1).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                b_states = states_t[batch_indices]
                b_actions = actions_t[batch_indices]
                b_old_log_probs = old_log_probs_t[batch_indices]
                b_returns = returns_t[batch_indices]
                b_advantages = advantages_t[batch_indices]

                log_probs, entropy, values = self.policy.evaluate_actions(b_states, b_actions)

                # Ratio for PPO clip objective
                ratios = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss & entropy loss
                value_loss = self.vf_coef * nn.MSELoss()(values, b_returns)
                entropy_loss = -self.beta_entropy * entropy.mean()

                total_loss = policy_loss + value_loss + entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)

    def load(self, filepath):
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
