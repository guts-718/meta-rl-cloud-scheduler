import random
from typing import List
from .base import BaseScheduler
from env.server import Server
from env.task import Task


class RandomScheduler(BaseScheduler):
    def select_action(self, servers: List[Server], queue: List[Task], current_time: int) -> int:
        if len(queue) == 0:
            return 0

        # random action including delay
        return random.randint(0, self.num_servers)