from env.task import Task
from env.server import Server

# Create server
server = Server(server_id=1)

# Create tasks
t1 = Task(cpu_req=0.3, ram_req=0.2, duration=5, arrival_time=0)
t2 = Task(cpu_req=0.5, ram_req=0.4, duration=3, arrival_time=0)

# Allocate
print("Allocating t1:", server.allocate(t1, current_time=0))
print("Allocating t2:", server.allocate(t2, current_time=0))

print(server)

# Simulate steps
for t in range(6):
    finished = server.step()
    print(f"\nTime {t+1}")
    print(server)
    print("Finished:", [task.id for task in finished])