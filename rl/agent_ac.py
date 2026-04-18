import torch
from torch.distributions import Categorical
from .actor_critic import ActorCritic


class ACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, entropy_coef=0.005):
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.reset_memory()

    def reset_memory(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)

        probs, value = self.model(state)
        dist = Categorical(probs)

        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        self.values.append(value.squeeze())
        self.entropies.append(dist.entropy())

        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        returns = []
        G = 0

        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        values = torch.stack(self.values)

        # Advantage
        advantages = returns - values.detach()

        # Actor loss
        actor_loss = -(torch.stack(self.log_probs) * advantages).mean()

        # Critic loss
        critic_loss = (returns - values).pow(2).mean()

        # Entropy bonus
        entropy_loss = -self.entropy_coef * torch.stack(self.entropies).mean()

        loss = actor_loss + 0.5 * critic_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        self.reset_memory()