from dataclasses import dataclass, field
import random
import itertools

_task_id_gen = itertools.count()


@dataclass
class Task:
    cpu_req: float          # normalized (0–1)
    ram_req: float          # normalized (0–1)
    duration: float         # total duration
    arrival_time: int

    id: int = field(default_factory=lambda: next(_task_id_gen))
    remaining_time: float = field(init=False)
    start_time: int | None = None

    def __post_init__(self):
        self.remaining_time = self.duration

    def step(self, time_delta: float = 1.0):
        """Simulate execution for one step."""
        if self.remaining_time > 0:
            self.remaining_time -= time_delta

    def is_finished(self) -> bool:
        return self.remaining_time <= 0

    def wait_time(self, current_time: int) -> float:
        return current_time - self.arrival_time