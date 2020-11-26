import json
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
import humanize
import argparse
import inquirer

parser = argparse.ArgumentParser(description="Visualize recorded memory data")
parser.add_argument('--input', type=str, help="input file (from mem_profile.py)", required=False, default=None)
parser.add_argument('--max_lines', type=int, help="Maximal number of lines to read", default=-1, required=False)
parser.add_argument('--customize', help="Select which processes to show", action="store_true")
parser.add_argument('--subtract', help='Subtract the baseline (memory at start)', action="store_true")


prefixes = ["ray::ImplicitFunc", "ray::RolloutWorker"]
args = parser.parse_args()

input_file = args.input
if args.input is None:
    files = sorted(os.listdir())[::-1]
    files = [x for x in files if x.startswith('mem_out') and x.endswith('.txt')]
    descrs = []
    dt_now = datetime.now()
    for fn in files:
        _, _, username, ts = fn.split('.')[0].split('_')
        dt_then = datetime.fromtimestamp(int(ts))
        delta = dt_now - dt_then
        descrs.append(f"User {username}, {delta} ago, {dt_then}")
    questions = [inquirer.List(
            'file',
                message="Which file to load?",
                choices=descrs)]
    answers = inquirer.prompt(questions)  # returns a dict
    choice = answers['file']
    index = descrs.index(choice)
    fn = files[descrs.index(choice)]
    input_file = fn

def get_lines():
    """Get lines from the log file as json strings."""
    global input_file
    f = open(input_file, 'r')
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

if args.subtract:
    for _, p in processes.items():
        values = [x['total_mem'] for x in p if x['total_mem'] > 0]

        for line in p:
            if values:
                line['total_mem'] -= min(values)

plt.figure(figsize=(10, 10))
plt.title(f"Memory usage for {input_file}")
for pid, p in processes.items():
    plt.plot([x['timestep'] for x in processes[pid]],
        [x['total_mem'] for x in processes[pid]], label=names[pid])
plt.legend()
plt.xlabel('time')
plt.ylabel('Mem, bytes')
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: humanize.naturalsize(x)))
plt.savefig(input_file + '.png', bbox_inches='tight')
plt.show()
