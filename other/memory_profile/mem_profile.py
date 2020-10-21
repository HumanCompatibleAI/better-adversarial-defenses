import psutil
import time
import getpass
import json

# Iterate over all running process
data = []
username = getpass.getuser()
time_start = int(time.time())
fn = f"mem_out_{username}_{time_start}.txt" 
print(f"Writing to {fn}")
f = open(fn, 'w')

while True:
    d = []
    for proc in psutil.process_iter():
        try:
            # Get process name & pid from process object.
            processName = proc.name()
            processID = proc.pid
            if proc.username() != username: continue
            d.append({'name': processName, 'id': processID, 'mem_info': proc.memory_info()._asdict(), 'timestep': time.time()})
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    data.append(d)
    time.sleep(1)
    f.write(json.dumps(d) + '\n')
