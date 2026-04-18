from env.cloud_env import CloudEnv
from schedulers.random_scheduler import RandomScheduler
from schedulers.first_fit import FirstFitScheduler
from schedulers.shortest_job import ShortestJobFirstScheduler

import numpy as np


def run_episode(env, scheduler, steps=500):
    state = env.reset()

    rewards = []
    utils = []
    queue_sizes = []
    completed = 0

    for _ in range(steps):
        action = scheduler.select_action(env.servers, list(env.queue), env.time)
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

    schedulers = {
        "Random": RandomScheduler(env.num_servers),
        "FirstFit": FirstFitScheduler(env.num_servers),
        "SJF": ShortestJobFirstScheduler(env.num_servers),
    }

    for name, scheduler in schedulers.items():
        result = run_episode(env, scheduler)

        print(f"\n{name} Results:")
        for k, v in result.items():
            print(f"{k}: {v:.4f}")