import torch
import copy
import numpy as np

from env.cloud_env import CloudEnv
from rl.agent_ac import ACAgent
from meta.tasks import get_task_set


def evaluate(env, agent, steps=300):
    state = env.reset()

    total_reward = 0
    total_completed = 0
    utils = []

    for _ in range(steps):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            probs, _ = agent.model(state_tensor)

            action = torch.distributions.Categorical(probs).sample().item()

        state, reward, _, info = env.step(action)

        total_reward += reward
        total_completed += info["completed_now"]
        utils.append(info["avg_util"])

    return {
        "reward": total_reward,
        "util": np.mean(utils),
        "completed": total_completed
    }


def adapt(agent, env, steps=15):
    state = env.reset()

    for _ in range(steps):
        action = agent.select_action(state)
        next_state, reward, _, _ = env.step(action)

        agent.store_reward(reward)
        state = next_state

    agent.update()


if __name__ == "__main__":
    tasks = get_task_set()

    env = CloudEnv(num_servers=5)
    state_dim = len(env.reset())
    action_dim = env.num_servers + 1

    # -------- Load Meta Model --------
    meta_agent = ACAgent(state_dim, action_dim)
    meta_agent.model.load_state_dict(torch.load("meta_model.pth"))

    # -------- Load RL Model --------
    rl_agent = ACAgent(state_dim, action_dim)
    # rl_agent.model.load_state_dict(torch.load("ac_model.pth"))

    print("\n=== META vs RL ADAPTATION TEST ===")

    for i, task in enumerate(tasks):
        print(f"\nTask {i+1}")

        env = CloudEnv(num_servers=5, task_config=task)

        # --- Copy agents ---
        meta_copy = copy.deepcopy(meta_agent)
        rl_copy = copy.deepcopy(rl_agent)

        # --- Before adaptation ---
        meta_before = evaluate(env, meta_copy)
        rl_before = evaluate(env, rl_copy)

        # --- Adapt ---
        adapt(meta_copy, env)
        adapt(rl_copy, env)

        # --- After adaptation ---
        meta_after = evaluate(env, meta_copy)
        rl_after = evaluate(env, rl_copy)

        print("META BEFORE:", meta_before)
        print("RL BEFORE:  ", rl_before)

        print("META AFTER :", meta_after)
        print("RL AFTER   :", rl_after)