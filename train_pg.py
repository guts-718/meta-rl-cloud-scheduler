from env.cloud_env import CloudEnv
from rl.agent import PGAgent

import torch
import numpy as np


env = CloudEnv(num_servers=5, seed=42)

state_dim = len(env.reset())
action_dim = env.num_servers + 1

agent = PGAgent(state_dim, action_dim)

EPISODES = 300
STEPS = 300

reward_history = []

for ep in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for _ in range(STEPS):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        agent.store_reward(reward)

        state = next_state
        total_reward += reward

    agent.update()
    reward_history.append(total_reward)

    if ep % 10 == 0:
        avg_last = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else total_reward
        print(f"Episode {ep}, Reward: {total_reward:.2f}, Avg(10): {avg_last:.2f}")

# Save model
torch.save(agent.policy.state_dict(), "pg_model.pth")
print("\nModel saved as pg_model.pth")