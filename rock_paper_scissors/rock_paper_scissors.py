import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from copy import deepcopy
from tqdm.notebook import tqdm
import tensorflow as tf
import numpy as np
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.compat.v1.enable_eager_execution()


# Rock Paper Scissors actions
action_to_descr = 'RPS'
n_act = len(action_to_descr)


def rewards(a1, a2):
    """Rock paper scissors game."""
    a1 = action_to_descr[a1]
    a2 = action_to_descr[a2]
    if a1 == a2:
        return (0, 0)
    outcomes = {'RP': (-1, 1),
                'RS': (1, -1),
                'PS': (-1, 1),

                }
    a1a2 = a1 + a2
    a2a1 = a1a2[::-1]
    if a1a2 in outcomes:
        return outcomes[a1a2]
    elif a2a1 in outcomes:
        return outcomes[a2a1][::-1]
    else:
        raise Exception("Unkown action pair %s" % a1a2)


def print_outcomes():
    for a1 in range(n_act):
        for a2 in range(n_act):
            r = rewards(a1, a2)
            ad1 = action_to_descr[a1]
            ad2 = action_to_descr[a2]
            descr = ""
            if r[0] > r[1]:
                descr = "%s wins" % ad1
            elif r[1] > r[0]:
                descr = "%s loses" % ad1
            else:
                descr = "tie"
            print("%s vs %s => %s" % (ad1, ad2, descr))
