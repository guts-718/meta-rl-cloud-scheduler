import copy
from rl.agent_ac import ACAgent


def train_on_task(env, base_model, steps=200):
    agent = copy.deepcopy(base_model)

    state = env.reset()

    for _ in range(steps):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        agent.store_reward(reward)
        state = next_state

    agent.update()

    return agent