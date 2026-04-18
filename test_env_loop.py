from env.cloud_env import CloudEnv
import random

env = CloudEnv(num_servers=5, seed=42)
state = env.reset()

print("Initial state dim:", len(state))

for step in range(20):
    # random policy for now
    action = random.randint(0, env.num_servers)  # 0..N

    next_state, reward, done, info = env.step(action)

    print(
        f"t={info['time']:02d} "
        f"a={action} "
        f"arr={info['arrivals']} "
        f"sch={info['scheduled_now']} "
        f"done={info['completed_now']} "
        f"q={info['queue_len']} "
        f"util={info['avg_util']:.2f} "
        f"r={reward:.3f}"
    )