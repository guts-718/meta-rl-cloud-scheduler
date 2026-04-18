from dataclasses import dataclass


@dataclass
class TaskConfig:
    lam: float
    cpu_range: tuple
    ram_range: tuple
    duration_range: tuple