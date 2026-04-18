from typing import List
from .base import BaseScheduler
from env.server import Server
from env.task import Task


class FirstFitScheduler(BaseScheduler):
    def select_action(self, servers: List[Server], queue: List[Task], current_time: int) -> int:
        if len(queue) == 0:
            return 0

        task = queue[0]

        for i, server in enumerate(servers):
            if server.can_allocate(task):
                return i + 1  # action space offset

        return 0  # delay if no server fits