from env.cloud_env import CloudEnv
from rl.agent import PGAgent

import numpy as np
import torch


env = CloudEnv(num_servers=5, seed=42)

state_dim = len(env.reset())
action_dim = env.num_servers + 1

agent = PGAgent(state_dim, action_dim)

EPISODES = 200
STEPS = 300

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

    if ep % 10 == 0:
        print(f"Episode {ep}, Reward: {total_reward:.2f}")

torch.save(agent.policy.state_dict(), "pg_model.pth")