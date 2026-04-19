import copy
from rl.agent_ac import ACAgent


def train_on_task(env, base_model, steps=300):
    agent = copy.deepcopy(base_model)

    state = env.reset()

    for step in range(steps):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        agent.store_reward(reward)
        state = next_state

    # Ensure at least some experience
    if len(agent.rewards) == 0:
        return agent

    for _ in range(3):   # multiple updates
        agent.update()
    return agent