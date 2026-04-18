from abc import ABC, abstractmethod
from typing import List
from env.server import Server
from env.task import Task


class BaseScheduler(ABC):
    def __init__(self, num_servers: int):
        self.num_servers = num_servers

    @abstractmethod
    def select_action(
        self,
        servers: List[Server],
        queue: List[Task],
        current_time: int
    ) -> int:
        """
        Returns:
            int → action
            0 = delay
            1..N = assign to server (id = action-1)
        """
        pass