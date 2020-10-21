import json
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
import humanize
import argparse
import inquirer

parser = argparse.ArgumentParser(description="Visualize recorded memory data")
parser.add_argument('--input', type=str, help="input file (from mem_profile.py)", required=True)
parser.add_argument('--max_lines', type=int, help="Maximal number of lines to read", default=-1, required=False)
parser.add_argument('--customize', help="Select which processes to show", action="store_true")


prefixes = ["ray::ImplicitFunc", "ray::RolloutWorker"]
args = parser.parse_args()


def get_lines():
    """Get lines from the log file as json strings."""
    f = open(args.input, 'r')
    for i, line in enumerate(f):
        if args.max_lines > 0 and i > args.max_lines:
            break
        yield json.loads(line)
    return f

# ask which processes to show
if args.customize:
    print("Reading the list of processes...")
    all_names = sorted({p['name'] for line in get_lines() for p in line})
    questions = [inquirer.Checkbox(
            'processes',
                message="Which processes to show?",
                choices=all_names)]
    answers = inquirer.prompt(questions)  # returns a dict
    prefixes = answers['processes']

# maps pid -> details
processes = {}

for line in get_lines():
    for p in line:
        if not any([p['name'].startswith(x) for x in prefixes]): continue
        p['timestep'] = datetime.fromtimestamp(p['timestep'])
        p['total_mem'] = p['mem_info']['rss']#np.sum(list(p['mem_info'].values()))
        if p['id'] not in processes:
            processes[p['id']] = []
        processes[p['id']].append(p)

names = {pid: p[-1]['name'] for pid, p in processes.items()}

plt.figure(figsize=(10, 10))
for pid, p in processes.items():
    plt.plot([x['timestep'] for x in processes[pid]],
        [x['mem_info']['rss'] for x in processes[pid]], label=names[pid])
plt.legend()
plt.xlabel('time')
plt.ylabel('Mem, bytes')
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: humanize.naturalsize(x)))
plt.savefig(args.input + '.png', bbox_inches='tight')
plt.show()
