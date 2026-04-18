from typing import List
from .base import BaseScheduler
from env.server import Server
from env.task import Task


class ShortestJobFirstScheduler(BaseScheduler):
    def select_action(self, servers: List[Server], queue: List[Task], current_time: int) -> int:
        if len(queue) == 0:
            return 0

        # pick shortest job from queue
        shortest_task = min(queue, key=lambda t: t.duration)

        for i, server in enumerate(servers):
            if server.can_allocate(shortest_task):
                return i + 1

        return 0