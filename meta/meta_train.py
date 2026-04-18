import torch
import copy
from env.cloud_env import CloudEnv
from meta.tasks import get_task_set
from meta.inner_loop import train_on_task
from rl.agent_ac import ACAgent


def meta_train():
    tasks = get_task_set()

    env = CloudEnv(num_servers=5)
    state_dim = len(env.reset())
    action_dim = env.num_servers + 1

    meta_agent = ACAgent(state_dim, action_dim)

    META_EPOCHS = 50

    for epoch in range(META_EPOCHS):
        meta_weights = []

        for task in tasks:
            env = CloudEnv(num_servers=5, task_config=task)

            adapted_agent = train_on_task(env, meta_agent)

            meta_weights.append(adapted_agent.model.state_dict())

        # -------- Meta Update (simple averaging) --------
        new_state_dict = copy.deepcopy(meta_agent.model.state_dict())

        for key in new_state_dict:
            new_state_dict[key] = torch.mean(
                torch.stack([w[key] for w in meta_weights]), dim=0
            )

        meta_agent.model.load_state_dict(new_state_dict)

        print(f"Meta Epoch {epoch} complete")

    torch.save(meta_agent.model.state_dict(), "meta_model.pth")

if __name__ == "__main__":
    meta_train()