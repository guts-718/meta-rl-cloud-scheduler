import torch
import numpy as np
from torch.distributions import Categorical

from .policy import PolicyNetwork


class PGAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, entropy_coef=0.01):
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.reset_memory()

    # ---------------- Memory ----------------

    def reset_memory(self):
        self.log_probs = []
        self.rewards = []
        self.entropies = []

    # ---------------- Action ----------------

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)

        probs = self.policy(state)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        self.log_probs.append(log_prob)
        self.entropies.append(entropy)

        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    # ---------------- Update ----------------

    def update(self):
        # 1. Compute discounted returns
        returns = []
        G = 0

        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)

        # 2. Normalize returns (stability)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 3. Baseline (mean return)
        baseline = returns.mean()

        # 4. Policy loss (with advantage)
        policy_loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            advantage = G - baseline
            policy_loss += -log_prob * advantage

        # 5. Entropy bonus (encourage exploration)
        entropy_loss = -self.entropy_coef * torch.stack(self.entropies).sum()

        # 6. Total loss
        loss = policy_loss + entropy_loss

        # 7. Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (stability)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)

        self.optimizer.step()

        # 8. Reset memory
        self.reset_memory()