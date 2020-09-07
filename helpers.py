import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import numbers


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

def fill_which_training(rdf):
    """Fill data on which player is being trained."""
    which_training_arr = []
    for _, line in rdf.iterrows():
        currently_training = set([p for p in policies if  not np.isnan(line[f"info/learner/{p}/total_loss"])])
        assert len(currently_training) == 1
        which_training = int(list(currently_training)[0].split('_')[1])
        which_training_arr.append(which_training)

    rdf['which_training'] = which_training_arr
    return rdf

def burst_sizes(lst):
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