from collections import deque
from typing import List, Tuple, Dict, Any

from .server import Server
from .task import Task
from .generator import TaskGenerator, TaskGenConfig


class CloudEnv:
    def __init__(
        self,
        num_servers: int = 10,
        cpu_capacity: float = 1.0,
        ram_capacity: float = 1.0,
        max_queue: int = 1000,
        seed: int | None = None,
    ):
        self.num_servers = num_servers
        self.cpu_capacity = cpu_capacity
        self.ram_capacity = ram_capacity
        self.max_queue = max_queue
        self.seed = seed

        self.servers: List[Server] = []
        self.queue: deque[Task] = deque()
        self.time: int = 0

        self.task_gen = TaskGenerator(TaskGenConfig(seed=seed))

        # stats
        self.total_completed = 0
        self.total_scheduled = 0
        self.total_dropped = 0

        self.task_gen = TaskGenerator(
            TaskGenConfig(
                lam=1.5,   # instead of default 2.0
                seed=seed
            )
        )

    # ---------------- Reset ----------------

    def reset(self) -> List[float]:
        self.servers = [
            Server(i, self.cpu_capacity, self.ram_capacity)
            for i in range(self.num_servers)
        ]
        self.queue.clear()
        self.time = 0

        self.total_completed = 0
        self.total_scheduled = 0
        self.total_dropped = 0

        # warm start: generate initial tasks
        self._arrivals()

        return self._get_state()

    # ---------------- Step ----------------

    def step(self, action: int) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        """
        action:
          0 -> delay
          1..N -> assign to server (id = action-1)
        """
        info = {}

        # 1) progress running tasks
        completed_now = 0
        for s in self.servers:
            finished = s.step()
            completed_now += len(finished)

        self.total_completed += completed_now

        # 2) new arrivals
        arrivals = self._arrivals()

        # 3) scheduling (FIFO head)
        scheduled_now = 0
        waiting_penalty = 0.0

        if len(self.queue) > 0:
            task = self.queue[0]  # peek

            if action == 0:
                # delay
                waiting_penalty = 0.01 * task.wait_time(self.time)
            else:
                sid = action - 1
                if 0 <= sid < self.num_servers:
                    ok = self.servers[sid].allocate(task, self.time)
                    if ok:
                        self.queue.popleft()
                        scheduled_now = 1
                        self.total_scheduled += 1
                    else:
                        # invalid allocation -> small penalty
                        waiting_penalty = 0.02
                else:
                    waiting_penalty = 0.02

        # 4) queue overflow handling (drop tail)
        dropped = 0
        while len(self.queue) > self.max_queue:
            self.queue.pop()
            dropped += 1
        self.total_dropped += dropped

        # 5) reward
        avg_util = self._avg_utilization()
        reward = (
            0.4 * avg_util
            + 0.4 * scheduled_now
            + 0.2 * completed_now
            - waiting_penalty
        )

        # 6) time++
        self.time += 1

        next_state = self._get_state()
        done = False  # continuing task

        info.update({
            "time": self.time,
            "arrivals": len(arrivals),
            "scheduled_now": scheduled_now,
            "completed_now": completed_now,
            "queue_len": len(self.queue),
            "avg_util": avg_util,
            "dropped": dropped,
        })

        return next_state, reward, done, info

    # ---------------- Internals ----------------

    def _arrivals(self) -> List[Task]:
        tasks = self.task_gen.sample_batch(self.time)
        for t in tasks:
            self.queue.append(t)
        return tasks

    def _avg_utilization(self) -> float:
        if self.num_servers == 0:
            return 0.0
        u = 0.0
        for s in self.servers:
            u += 0.5 * (s.cpu_utilization() + s.ram_utilization())
        return u / self.num_servers

    def _get_state(self) -> List[float]:
        """
        Fixed-size state:
        [cpu_util_1, ram_util_1, ..., cpu_util_N, ram_util_N,
         task_cpu, task_ram, task_duration_norm, task_wait_norm]
        """
        state: List[float] = []

        # servers
        for s in self.servers:
            state.append(s.cpu_utilization())
            state.append(s.ram_utilization())

        # head-of-line task (FIFO)
        if len(self.queue) > 0:
            t = self.queue[0]
            state.extend([
                t.cpu_req,
                t.ram_req,
                min(1.0, t.duration / 50.0),          # normalize
                min(1.0, t.wait_time(self.time) / 50.0),
            ])
        else:
            state.extend([0.0, 0.0, 0.0, 0.0])

        return state