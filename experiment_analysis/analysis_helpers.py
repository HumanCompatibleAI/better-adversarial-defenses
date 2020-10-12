from ray.tune.analysis import Analysis
import pandas as pd
import multiprocessing
from make_video import make_video, parser
import ray
import shutil
from IPython.display import display, FileLink
import os

def make_video_parallel(checkpoints, arguments):
    """Run make_video in parallel.
    
    Args:
        checkpoints: list of checkpoints to process
        arguments: list of strings to supply to the parser
        
    Returns:
        List of results from make_video
    
    """
    make_video_remote = ray.remote(make_video)
    args = [parser.parse_args(['--checkpoint', ckpt, *arguments]) for ckpt in checkpoints]
    res = [make_video_remote.remote(a) for a in args]
    res = ray.get(res)
    return res

def get_videos(df, steps=2, load_normal=False, display=':0'):
    """Add video column to the dataframe."""
    args = ['--steps', str(steps), '--display', str(display)]
    if load_normal:
        args += ['--load_normal', 'True']
    res = make_video_parallel(list(df.checkpoint_rllib), args)
    r = [r['video'] for r in res]
    return r

def add_scores(df, steps=200, load_normal=False):
    """Compute scores w.r.t. all opponents."""
    # computing score with the normal opponent
    make_video_remote = ray.remote(make_video)
    args = ['--steps', str(steps), '--no_video', 'True']
    if load_normal:
        args += ['--load_normal', 'True']
    res = make_video_parallel(list(df.checkpoint_rllib), args)
    return res


def download_videos(list_of_videos, list_of_filenames):
    """Download videos after re-naming them."""
    # download videos
    shutil.rmtree('videos/', ignore_errors=True)
    os.makedirs('videos', exist_ok=True)

    for video, fn in zip(list_of_videos, list_of_filenames):
        if not fn.endswith('.mp4'):
            fn = fn + '.mp4'
        shutil.copyfile(video, f"videos/{fn}")

    os.system('zip -r videos.zip videos')
    local_file = FileLink('videos.zip', result_html_prefix="Download videos: ")
    display(local_file)