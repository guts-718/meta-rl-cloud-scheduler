import torch
import numpy as np
from torch.distributions import Categorical

from .policy import PolicyNetwork


class PGAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99):
        self.gamma = gamma

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.reset_memory()

    def reset_memory(self):
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)

        probs = self.policy(state)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.log_probs.append(log_prob)

        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        returns = []
        G = 0

        # compute discounted returns
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)

        # normalize returns (important)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            loss += -log_prob * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.reset_memory()