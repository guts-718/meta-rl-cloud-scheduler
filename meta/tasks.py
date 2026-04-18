from env.task_configs import TaskConfig


def get_task_set():
    return [
        TaskConfig(1.0, (0.1, 0.4), (0.1, 0.4), (2, 10)),
        TaskConfig(2.0, (0.2, 0.6), (0.2, 0.6), (5, 20)),
        TaskConfig(0.5, (0.05, 0.3), (0.05, 0.3), (1, 5)),
        TaskConfig(3.0, (0.3, 0.8), (0.3, 0.8), (10, 30)),
        TaskConfig(1.5, (0.1, 0.5), (0.1, 0.5), (3, 15)),
    ]