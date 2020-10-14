#!/usr/bin/env python
# coding: utf-8

# In[17]:


from ray.tune.analysis import Analysis
import pandas as pd
from ap_rllib.make_video import make_video, parser
import ray
from matplotlib import pyplot as plt
from ap_rllib.helpers import get_df_from_logdir, burst_sizes, fill_which_training, iterate_bursts
import numpy as np
import matplotlib.colors as mcolors
import pickle


# In[2]:


# need ray for parallel evaluation
ray.shutdown()
ray.init(num_cpus=28, ignore_reinit_error=True, log_to_driver=False)


# In[3]:


# loading data
exp_name = "adversarial_tune_bursts_exp_sb"
config = "bursts_exp_sb"
analysis = Analysis("/home/sergei/ray_results/" + exp_name)
df = analysis.dataframe(metric='policy_reward_mean/player_1', mode=None)
df = df[df.episodes_total > 100000]

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


# In[8]:


@ray.remote(max_calls=1)
def make_video_1(*args, **kwargs):
    return make_video(*args, **kwargs)

def make_video_parallel(checkpoints, arguments):
    """Run make_video in parallel.
    
    Args:
        checkpoints: list of checkpoints to process
        arguments: list of strings to supply to the parser
        
    Returns:
        List of results from make_video
    
    """
    args = [parser.parse_args(['--checkpoint', ckpt, *arguments]) for ckpt in checkpoints]
    res = [make_video_1.remote(a) for a in args]
    res = ray.get(res)
    return res

def get_videos(df, steps=2, load_normal=False, display=':0', config=None):
    """Add video column to the dataframe."""
    args = ['--steps', str(steps), '--display', str(display), '--config', str(config)]
    if load_normal:
        args += ['--load_normal', 'True']
    res = make_video_parallel(list(df.checkpoint_rllib), args)
    r = [r['video'] for r in res]
    return r

def get_scores(df, steps=200, load_normal=False, config=None):
    """Compute scores w.r.t. all opponents."""
    # computing score with the normal opponent
    make_video_remote = ray.remote(make_video)
    args = ['--steps', str(steps), '--no_video', 'True', '--config', str(config)]
    if load_normal:
        args += ['--load_normal', 'True']
    res = make_video_parallel(list(df.checkpoint_rllib), args)
    return res


# In[23]:


def process_trial_dataframe(rdf, fn="experiment"):


    # List of all players
    reward_prefix = 'policy_reward_mean/'
    POLICIES = [x[len(reward_prefix):] for x in rdf.columns if x.startswith(reward_prefix)]
    print("Policies:", POLICIES)

    exponent = rdf['config/_burst_exponent'][0]
    print("Exponent:", exponent)

    # filling the 'which_training' column
    rdf = fill_which_training(rdf, POLICIES)
    burst_sizes_ = burst_sizes(rdf['which_training'])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Which player is trained?")
    plt.plot(rdf['timesteps_total'], rdf['which_training'])
    plt.xlabel('Timesteps')
    plt.subplot(1, 2, 2)
    plt.title(f"Burst sizes with exponent {round(exponent, 2)}")
    plt.plot(burst_sizes_)
    plt.xlabel('Burst number')
    plt.ylabel('Burst size')
    plt.savefig(fn + "_bursts.png", bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.xlabel('Timesteps')

    fields = [x for x in rdf.columns if x.startswith(reward_prefix)]

    iterate_bursts(rdf, burst_line_plot, state={'fields': fields,
                                                'colors': {f: cols[i] for i, f in enumerate(fields)},
                                                'rolling': 10})

    plt.legend()
    plt.savefig(fn + "_reward.png", bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.xlabel('Timesteps')

    fields = [x for x in rdf.columns if x.endswith('value_loss')]

    iterate_bursts(rdf, burst_line_plot, state={'fields': fields,
                                                'colors': {f: cols[i] for i, f in enumerate(fields)},
                                                'rolling': 10})

    plt.legend()
    plt.savefig(fn + "_loss.png", bbox_inches='tight')
    plt.show()


    # was there a switch?
    rdf['switch'] = list(np.array(rdf['which_training'].iloc[1:]) != rdf['which_training'].iloc[:-1]) + [True]

    rdf_e = rdf[rdf['switch'] == 1]
    print("To evaluate", len(rdf_e))

    # obtaining the scores


    scores_normal = get_scores(rdf_e, steps=SCORE_STEPS, load_normal=True, config=config)
    scores = get_scores(rdf_e, steps=SCORE_STEPS, load_normal=False, config=config)

    POLICIES_SHOW = list(POLICIES)
    if len(POLICIES_SHOW) == 2:
        POLICIES_SHOW = POLICIES_SHOW[0:1]

    # plotting the win rate
    plt.title("Win rate")
    for p in POLICIES_SHOW:
        plt.plot(rdf_e['timesteps_total'], [x[f"wins_policy_{p}_reward"] for x in scores], label=p + '_adversarial')
        plt.scatter(rdf_e['timesteps_total'][0], [x[f"wins_policy_{p}_reward"] for x in scores][0])
        plt.plot(rdf_e['timesteps_total'], [x[f"wins_policy_{p}_reward"] for x in scores_normal], label=p + '_normal')
        plt.scatter(rdf_e['timesteps_total'][0], [x[f"wins_policy_{p}_reward"] for x in scores_normal][0])
        assert all([x[f"ties_policy_{p}_reward"] == 0 for x in scores])
    plt.xlabel("Time-steps")

    iterate_bursts(rdf, burst_background, state={'min_metric': 0, 'max_metric': 100})

    plt.legend()
    plt.savefig(fn + "_win_rate.png", bbox_inches='tight')

    plt.show()

    videos_normal = get_videos(rdf_e, steps=VIDEO_STEPS, load_normal=True, config=config)
    videos = get_videos(rdf_e, steps=VIDEO_STEPS, load_normal=False, config=config)

    result = {'scores': scores, 'scores_normal': scores_normal,
              'videos': videos, 'videos_normal': videos_normal,
              'xs_e': rdf_e['timesteps_total'],
              'policies': POLICIES,
              'burst_sizes': burst_sizes_,
              'rdf': rdf,
              'rdf_e': rdf_e
             }
    
    return result

SCORE_STEPS = 15
VIDEO_STEPS = 15
fns = []

for i, trial in df.iterrows():
    rdf = get_df_from_logdir(trial.logdir)
    fn = f"{exp_name}_trial_{str(i)}.pkl"
    res = process_trial_dataframe(rdf, fn)
    
    pickle.dump(res, open(fn, "wb"))
    print(fn)
    fns.append(fn)

print("All", fns)
