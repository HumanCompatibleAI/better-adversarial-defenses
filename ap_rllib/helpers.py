import json
import gym
import numbers
import os

import numpy as np
import pandas as pd
import ray
import ray.tune as tune
from tqdm import tqdm
import multiprocessing
import logging


def ray_init(shutdown=True, tmp_dir='/tmp', **kwargs):
    """Initialize ray."""
    # number of CPUs on the machine
    num_cpus = multiprocessing.cpu_count()

    # restart ray / use existing session
    if shutdown:
        ray.shutdown()
    if not shutdown:
        kwargs['ignore_reinit_error'] = True

    # if address is not known, launch new instance
    if 'address' not in kwargs:
        # pretending we have more so that workers are never stuck
        # resources are limited by `tune_cpu` resources that we create
        kwargs['num_cpus'] = num_cpus * 2

        # `tune_cpu` resources are used to limit number of
        # concurrent trials
        kwargs['resources'] = {'tune_cpu': num_cpus}
        kwargs['temp_dir'] = tmp_dir

    # only showing errors, to prevent too many messages from coming
    kwargs['logging_level'] = logging.ERROR

    # launching ray
    return ray.init(log_to_driver=True, **kwargs)


def save_gym_space(space):
    """Serialize gym.space."""
    if isinstance(space, gym.spaces.Box):
        low = space.low.flatten()[0]
        high = space.high.flatten()[0]
        return dict(type_='Box', low=low, high=high,
                    shape=space.shape, dtype=space.dtype)
    raise TypeError(f"Type {type(space)} {space} is unsupported")
    
def load_gym_space(d):
    """Load gym.space from save_gym_space result."""
    assert isinstance(d, dict)
    if d['type_'] == 'Box':
        return gym.spaces.Box(low=d['low'], high=d['high'],
                              shape=d['shape'], dtype=d['dtype'])
    raise TypeError(f"Type {d['type_']} is unsupported")

def dict_to_sacred(ex, d, iteration, prefix=''):
    """Log a dictionary to sacred."""
    for k, v in d.items():
        if isinstance(v, dict):
            dict_to_sacred(ex, v, iteration, prefix=prefix + k + '/')
        elif isinstance(v, float) or isinstance(v, int):
            ex.log_scalar(prefix + k, v, iteration)


def tune_compose(obj, f):
    """Apply f after sampling from obj."""
    return tune.sample_from(lambda x: f(obj.func(x)))


def tune_int(obj):
    """Convert result to int after sampling from obj."""
    return tune_compose(obj, round)


def sample_int(obj):
    """Convert tune distribution to integer, backward-compatible name."""
    return tune_int(obj)

class Unpickleable(object):
    """Represent an unpickleable object"""
    def __init__(self, obj):
        self.obj_type = str(type(obj))
        self.obj_str = str(obj)
    def __repr__(self):
        return f"<Unpickleable({self.obj_str}, type={self.obj_type}>"

def filter_pickleable(d):
    """Recursively keep only pickleable objects."""
    basic_types = {int, float, bool, str, np.ndarray, type(None)}
    if type(d) in basic_types:
        return d
    elif isinstance(d, tuple):
        return tuple(filter_pickleable(z) for z in d)
    elif isinstance(d, list):
        return [filter_pickleable(z) for z in d]
    elif isinstance(d, set):
        return {filter_pickleable(z) for z in d}
    elif isinstance(d, dict):
        return {filter_pickleable(x): filter_pickleable(y) for x, y in d.items()}
    else:
        return Unpickleable(d)


def dict_get_any_value(d):
    """Return any value of a dict."""
    return list(d.values())[0]


def unlink_ignore_error(p):
    """Unlink without complaining if the file does not exist."""
    try:
        os.unlink(p)
    except:
        pass


def flatten_dict_keys(dct, prefix='', separator='/'):
    """Nested dictionary to a flat dictionary."""
    result = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            subresult = flatten_dict_keys(value, prefix=prefix + key + '/',
                                          separator=separator)
            result.update(subresult)
        else:
            result[prefix + key] = value
    return result


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


def fill_which_training(rdf, policies):
    """Fill data on which player is being trained."""
    which_training_arr = []
    for _, line in rdf.iterrows():
        currently_training = set([p for p in policies if not np.isnan(line[f"info/learner/{p}/policy_loss"])])
        assert len(currently_training) == 1
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

def iterate_bursts(rdf, target, min_size=5, state=None, target_field = 'which_training'):
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