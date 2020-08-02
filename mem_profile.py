import psutil
import time
import json

# Iterate over all running process
data = []
f = open('mem_out_%d.txt' % int(time.time()), 'w')

while True:
    d = []
    for proc in psutil.process_iter():
        try:
            # Get process name & pid from process object.
            processName = proc.name()
            processID = proc.pid
            print(processName , ' ::: ', processID, proc.memory_info())
            d.append({'name': processName, 'id': processID, 'mem_info': proc.memory_info()._asdict(), 'timestep': time.time()})
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    data.append(d)
    time.sleep(1)
    f.write(json.dumps(d) + '\n')
