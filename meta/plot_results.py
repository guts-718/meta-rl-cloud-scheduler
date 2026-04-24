import torch
import copy
import numpy as np
import matplotlib.pyplot as plt

from env.cloud_env import CloudEnv
from rl.agent_ac import ACAgent
from meta.tasks import get_task_set


# -----------------------------
# Collect reward over time
# -----------------------------
def get_learning_curve(agent, env, steps=50):
    state = env.reset()

    rewards = []

    for _ in range(steps):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            probs, _ = agent.model(state_tensor)

            action = torch.distributions.Categorical(probs).sample().item()

        state, reward, _, _ = env.step(action)
        rewards.append(reward)

    return np.cumsum(rewards)  # cumulative reward


# -----------------------------
# Adapt agent
# -----------------------------
def adapt(agent, env, steps=20):
    state = env.reset()

    for _ in range(steps):
        action = agent.select_action(state)
        next_state, reward, _, _ = env.step(action)

        agent.store_reward(reward)
        state = next_state

    agent.update()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    tasks = get_task_set()

    env = CloudEnv(num_servers=5)
    state_dim = len(env.reset())
    action_dim = env.num_servers + 1

    # Load models
    meta_agent = ACAgent(state_dim, action_dim)
    meta_agent.model.load_state_dict(torch.load("meta_model.pth"))

    rl_agent = ACAgent(state_dim, action_dim)
    rl_agent.model.load_state_dict(torch.load("ac_model.pth"))

    # -------- Plot 1: Adaptation Curve --------
    for i, task in enumerate(tasks):
        env = CloudEnv(num_servers=5, task_config=task)

        meta_copy = copy.deepcopy(meta_agent)
        rl_copy = copy.deepcopy(rl_agent)

        # Adapt both
        adapt(meta_copy, env, steps=20)
        adapt(rl_copy, env, steps=20)

        meta_curve = get_learning_curve(meta_copy, env)
        rl_curve = get_learning_curve(rl_copy, env)

        plt.figure()
        plt.plot(meta_curve, label="Meta-RL")
        plt.plot(rl_curve, label="RL")

        plt.title(f"Adaptation Curve - Task {i+1}")
        plt.xlabel("Steps")
        plt.ylabel("Cumulative Reward")
        plt.legend()

        plt.savefig(f"task_{i+1}_curve.png")
        plt.close()

    print("Adaptation curves saved.")


    # -------- Plot 2: Final Comparison --------
    meta_scores = []
    rl_scores = []

    for task in tasks:
        env = CloudEnv(num_servers=5, task_config=task)

        meta_copy = copy.deepcopy(meta_agent)
        rl_copy = copy.deepcopy(rl_agent)

        adapt(meta_copy, env, steps=20)
        adapt(rl_copy, env, steps=20)

        meta_reward = np.sum(get_learning_curve(meta_copy, env))
        rl_reward = np.sum(get_learning_curve(rl_copy, env))

        meta_scores.append(meta_reward)
        rl_scores.append(rl_reward)

    x = np.arange(len(tasks))

    plt.figure()
    plt.bar(x - 0.2, meta_scores, width=0.4, label="Meta-RL")
    plt.bar(x + 0.2, rl_scores, width=0.4, label="RL")

    plt.xlabel("Tasks")
    plt.ylabel("Total Reward")
    plt.title("Meta vs RL Performance")
    plt.legend()

    plt.savefig("comparison.png")
    plt.close()

    print("Comparison plot saved.")