from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors


def burst_background(rdf, trained_now, accumulator, rolling=10, state=None):
    """Add green-red bursts background."""
    min_metric, max_metric = state['min_metric'], state['max_metric']
    
    xs = accumulator['timesteps_total']
    
    colors_bg = {1: "red", 2: "green"}
    labels_bg = {1: "Training opponent", 2: "Training victim"}    
    
    plt.fill_between(xs, min_metric, max_metric, alpha=0.1, color=colors_bg[trained_now],
                     label=labels_bg[trained_now] if trained_now not in state else None)
    state[trained_now] = True
    
    return state


def burst_line_plot(rdf, trained_now, accumulator, rolling=10, state=None):
    """Plot lines from accumulator."""
    
    fields, colors, rolling = state['fields'], state['colors'], state['rolling']
    
    
    all_ys = {field: rdf[field] for field in fields}
    all_ys_roll = {f: pd.Series(all_ys[f]).rolling(rolling) for f in fields}
    all_ys_mean = {f: all_ys_roll[f].mean() for f in fields}
    all_ys_std = {f: all_ys_roll[f].std() for f in fields}

    min_metric = np.min([np.min(all_ys[f]) for f in fields])
    max_metric = np.max([np.max(all_ys[f]) for f in fields])

    
    xs = accumulator['timesteps_total']
    
    colors_bg = {1: "red", 2: "green"}
    labels_bg = {1: "Training opponent", 2: "Training victim"}

    
    for f in fields:
        ys = all_ys_mean[f][min(accumulator.index):max(accumulator.index) + 1]
        ys_std = all_ys_std[f][min(accumulator.index):max(accumulator.index) + 1]


        plt.plot(xs, ys, color=colors[f], alpha=1, label=f if 'legend_y' not in state else None)
        plt.fill_between(xs, ys - 3 * ys_std, ys + 3 * ys_std, color=colors[f], alpha=0.3)
    state['legend_y'] = True
    
    
    plt.fill_between(xs, min_metric, max_metric, alpha=0.1, color=colors_bg[trained_now],
                     label=labels_bg[trained_now] if trained_now not in state else None)
    state[trained_now] = True
    
    return state

cols = [x for x in mcolors.TABLEAU_COLORS]


def fill_which_training(rdf, policies):
    """Fill data on which player is being trained."""
    which_training_arr = []
    for i, line in rdf.iterrows():
        currently_training = set([p for p in policies if not np.isnan(line.get(f"info/learner/{p}/policy_loss", np.nan))])
        assert len(currently_training) == 1, f"Training both at step {i}: {currently_training}"
        which_training = int(list(currently_training)[0].split('_')[1])
        which_training_arr.append(which_training)

    rdf['which_training'] = which_training_arr
    return rdf


def burst_sizes(wt):
    """Get burst sizes from a list of players being trained at iterations."""
    prev_val = 0
    current = 0
    arr = []
    for val in wt:
        current += 1
        if prev_val != val:
            arr.append(current)
            current = 0
        prev_val = val
    return arr

def iterate_bursts(rdf, target, min_size=5, state=None, target_field='which_training'):
    """Iterate a function over all bursts."""
    seen_data = 0

    #with tqdm(total=len(rdf)) as pbar:
    while seen_data < len(rdf):
        accumulator = []
        for i in range(seen_data, len(rdf)):
            if (rdf.iloc[i][target_field] == rdf.iloc[seen_data][target_field]) or len(accumulator) < min_size:
                accumulator.append(rdf.iloc[i])
            else:
                break
        accumulator = pd.DataFrame(accumulator)
                
        state = target(rdf=rdf, trained_now=accumulator.iloc[0][target_field],
                       accumulator=accumulator, state=state)

        seen_data += len(accumulator)
            #pbar.update(len(accumulator))