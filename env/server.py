from typing import List
from .task import Task


class Server:
    def __init__(self, server_id: int, cpu_capacity: float = 1.0, ram_capacity: float = 1.0):
        self.id = server_id

        self.cpu_capacity = cpu_capacity
        self.ram_capacity = ram_capacity

        self.cpu_used = 0.0
        self.ram_used = 0.0

        self.running_tasks: List[Task] = []

    # ---------- Capacity Checks ----------

    def can_allocate(self, task: Task) -> bool:
        return (
            self.cpu_used + task.cpu_req <= self.cpu_capacity
            and self.ram_used + task.ram_req <= self.ram_capacity
        )

    # ---------- Allocation ----------

    def allocate(self, task: Task, current_time: int) -> bool:
        if not self.can_allocate(task):
            return False

        self.cpu_used += task.cpu_req
        self.ram_used += task.ram_req

        task.start_time = current_time
        self.running_tasks.append(task)

        return True

    # ---------- Simulation Step ----------

    def step(self, time_delta: float = 1.0):
        finished_tasks = []

        for task in self.running_tasks:
            task.step(time_delta)
            if task.is_finished():
                finished_tasks.append(task)

        # Remove finished tasks
        for task in finished_tasks:
            self.running_tasks.remove(task)
            self.cpu_used -= task.cpu_req
            self.ram_used -= task.ram_req

        return finished_tasks

    # ---------- Metrics ----------

    def cpu_utilization(self) -> float:
        return self.cpu_used / self.cpu_capacity

    def ram_utilization(self) -> float:
        return self.ram_used / self.ram_capacity

    def __repr__(self):
        return (
            f"Server(id={self.id}, "
            f"cpu={self.cpu_used:.2f}/{self.cpu_capacity}, "
            f"ram={self.ram_used:.2f}/{self.ram_capacity}, "
            f"tasks={len(self.running_tasks)})"
        )
    
    def cpu_utilization(self) -> float:
        return min(1.0, max(0.0, self.cpu_used / self.cpu_capacity))

    def ram_utilization(self) -> float:
        return min(1.0, max(0.0, self.ram_used / self.ram_capacity))
    
    def debug_state(self):
        return {
            "id": self.id,
            "cpu_used": round(self.cpu_used, 3),
            "ram_used": round(self.ram_used, 3),
            "tasks": len(self.running_tasks),
        }