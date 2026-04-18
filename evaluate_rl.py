from env.cloud_env import CloudEnv
from schedulers.first_fit import FirstFitScheduler
from schedulers.random_scheduler import RandomScheduler
from rl.agent import PGAgent

import torch
import numpy as np


def evaluate_policy(env, agent=None, scheduler=None, steps=500):
    state = env.reset()

    rewards = []
    utils = []
    queue_sizes = []
    completed = 0

    for _ in range(steps):
        # -------- Select Action --------
        if agent:
            with torch.no_grad():  # IMPORTANT: no training during eval
                action = agent.select_action(state)
        else:
            action = scheduler.select_action(env.servers, list(env.queue), env.time)

        # -------- Step Environment --------
        state, reward, done, info = env.step(action)

        rewards.append(reward)
        utils.append(info["avg_util"])
        queue_sizes.append(info["queue_len"])
        completed += info["completed_now"]

    return {
        "avg_reward": np.mean(rewards),
        "avg_util": np.mean(utils),
        "avg_queue": np.mean(queue_sizes),
        "total_completed": completed
    }


if __name__ == "__main__":
    env = CloudEnv(num_servers=5, seed=42)

    state_dim = len(env.reset())
    action_dim = env.num_servers + 1

    # -------- Load Trained RL Agent --------
    agent = PGAgent(state_dim, action_dim)

    agent.policy.load_state_dict(torch.load("pg_model.pth"))
    agent.policy.eval()

    # -------- Baselines --------
    schedulers = {
        "Random": RandomScheduler(env.num_servers),
        "FirstFit": FirstFitScheduler(env.num_servers),
    }

    print("\n--- Baselines ---")
    for name, scheduler in schedulers.items():
        result = evaluate_policy(env, scheduler=scheduler)
        print(f"\n{name}:")
        for k, v in result.items():
            print(f"{k}: {v:.4f}")

    # -------- RL Agent --------
    print("\n--- RL Agent ---")
    result = evaluate_policy(env, agent=agent)
    for k, v in result.items():
        print(f"{k}: {v:.4f}")