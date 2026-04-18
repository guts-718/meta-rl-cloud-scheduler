from dataclasses import dataclass
import numpy as np
import random
from .task import Task


@dataclass
class TaskGenConfig:
    lam: float = 2.0                 # avg tasks per step (Poisson λ)
    cpu_range: tuple = (0.05, 0.6)   # normalized
    ram_range: tuple = (0.05, 0.6)
    duration_range: tuple = (2, 20)  # time steps
    seed: int | None = None


class TaskGenerator:
    def __init__(self, cfg: TaskGenConfig):
        self.cfg = cfg
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
            random.seed(cfg.seed)

    def sample_task(self, current_time: int) -> Task:
        cpu = random.uniform(*self.cfg.cpu_range)
        ram = random.uniform(*self.cfg.ram_range)

        # add slight stochasticity: skewed durations (more short, some long)
        base = random.uniform(*self.cfg.duration_range)
        if random.random() < 0.2:   # 20% long jobs
            base *= random.uniform(1.5, 2.5)

        duration = max(1.0, base)

        return Task(
            cpu_req=min(1.0, cpu),
            ram_req=min(1.0, ram),
            duration=duration,
            arrival_time=current_time
        )

    def sample_batch(self, current_time: int):
        k = np.random.poisson(self.cfg.lam)
        return [self.sample_task(current_time) for _ in range(k)]