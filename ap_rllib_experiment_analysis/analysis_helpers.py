from ap_rllib.make_video import make_video, parser
from ap_rllib.helpers import flatten_dict_keys
import ray
import shutil
from IPython.display import display, FileLink
import os
from os.path import expanduser
from tqdm import tqdm
import json
import numbers
import numpy as np
import pandas as pd
import gin


def get_last_checkpoint(config_name):
    from ap_rllib.config import CONFIGS
    """Get last checkpoint for an experiment."""
    home = expanduser("~")
    trial_name = CONFIGS[config_name]['_call']['name']
    path = os.path.join(home, 'ray_results', trial_name)
    trial = sorted(os.listdir(path))[-1]
    path_with_trial = os.path.join(path, trial)
    df = get_df_from_logdir(path_with_trial, do_tqdm=False)
    checkpoint = str(df.checkpoint_rllib.iloc[-1])
    return checkpoint

@ray.remote(max_calls=1)
def make_video_1(*args, **kwargs):
    return make_video(*args, **kwargs)

def ray_get_tqdm(remaining_ids):
    """Get results from a list of remote ids, with progress updates."""
    
    orig_ids = remaining_ids
    
    with tqdm(total=len(remaining_ids)) as pbar:
        while remaining_ids:
            ready_ids, remaining_ids_new = ray.wait(remaining_ids)
            if len(remaining_ids_new) < len(remaining_ids):
                pbar.update(len(remaining_ids) - len(remaining_ids_new))
            remaining_ids = remaining_ids_new

    return ray.get(orig_ids)

def make_video_parallel(checkpoints, arguments):
    """Run make_video in parallel.
    
    Args:
        checkpoints: list of checkpoints to process
        arguments: list of strings to supply to the parser
        
    Returns:
        List of results from make_video
    
    """
    args = [parser.parse_args(['--checkpoint', ckpt, *arguments]) for ckpt in checkpoints]
    remaining_ids = [make_video_1.remote(a) for a in args]
    return ray_get_tqdm(remaining_ids)


@gin.configurable
def get_videos(df, steps=2, load_normal=False, display=':0', config=None):
    """Add video column to the dataframe."""
    args = ['--steps', str(steps), '--display', str(display), '--config', str(config)]
    if load_normal:
        args += ['--load_normal', 'True']
    res = make_video_parallel(list(df.checkpoint_rllib), args)
    r = [r['video'] for r in res]
    return r


@gin.configurable
def get_scores(df, steps=200, load_normal=False, config=None):
    """Compute scores w.r.t. all opponents."""
    # computing score with the normal opponent
    make_video_remote = ray.remote(make_video)
    args = ['--steps', str(steps), '--no_video', 'True', '--config', str(config)]
    if load_normal:
        args += ['--load_normal', 'True']
    res = make_video_parallel(list(df.checkpoint_rllib), args)
    return res

def concat_videos(videos, out_fn):
    """Concatenate videos into one"""
    list_fn = "list_file.txt"
    with open(list_fn, 'w') as f:
        for video in videos:
            f.write(f"file {video}\n")
    os.system(f"ffmpeg -f concat -safe 0 -i {list_fn} -c copy {out_fn}.mp4")
    os.unlink(list_fn)

class VideosDownloader(object):
    """Download videos after re-naming them."""
    def __enter__(self):
        shutil.rmtree('videos/', ignore_errors=True)
        os.makedirs('videos', exist_ok=True)
        return self
        
    def __exit__(self, type_, value, traceback):
        os.system('zip -r videos.zip videos')
        local_file = FileLink('videos.zip', result_html_prefix="Download videos: ")
        display(local_file)
        
    def add_video(self, source, dest_filename):
        if not dest_filename.endswith('.mp4'):
            dest_filename = dest_filename + '.mp4'
        shutil.copyfile(source, f"videos/{dest_filename}")
        
        
def read_json_array_flat(logdir):
    """Read results.json from tune logdir."""
    data_current = []
    for line in open(logdir + '/result.json', 'r').readlines():
        data = json.loads(line)
        dict_current = flatten_dict_keys(data)
        data_current.append(dict_current)
    return data_current


def get_df_from_logdir(logdir, do_tqdm=True):
    """Obtain a dataframe from tune logdir."""

    # obtaining data
    data_current = read_json_array_flat(logdir)

    # list of array statistics
    array_stats = {'mean': np.mean, 'max': np.max, 'min': np.min, 'std': np.std, 'median': np.median}

    # computing statistics for arrays
    for line in (tqdm if do_tqdm else (lambda x: x))(data_current):
        for key in list(line.keys()):
            if not isinstance(line[key], list): continue
            if not line[key]: continue
            if not all([isinstance(x, numbers.Number) for x in line[key]]): continue
            for name, fcn in array_stats.items():
                try:
                    line[key + '_' + name] = fcn(line[key])
                except Exception as e:
                    print(e, name, key, fcn, line[key])
                    raise e

    # list of all keys
    all_keys = set()
    for line in data_current:
        all_keys.update(line.keys())

    # adding keys with None values
    for line in data_current:
        for key in all_keys:
            if key not in line:
                line[key] = None

    df_current = pd.DataFrame(data_current)

    return df_current