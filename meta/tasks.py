from env.task_configs import TaskConfig


def get_task_set():
    return [
        TaskConfig(1.0, (0.1, 0.5), (0.1, 0.5), (2, 15)),
        TaskConfig(1.5, (0.1, 0.5), (0.1, 0.5), (3, 18)),
        TaskConfig(2.0, (0.1, 0.6), (0.1, 0.6), (5, 20)),
    ]