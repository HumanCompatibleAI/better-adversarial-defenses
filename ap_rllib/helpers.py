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

    # add webui
    kwargs['include_webui'] = True

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
