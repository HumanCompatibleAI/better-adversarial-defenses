from copy import deepcopy
from functools import partial

import numpy as np
from ray import tune
from ray.rllib.agents.es import ESTrainer, ESTFPolicy
from ray.rllib.agents.ppo import APPOTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.appo_tf_policy import AsyncPPOTFPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune.schedulers import ASHAScheduler

from frankenstein.remote_trainer import ExternalTrainer
from gym_compete_rllib import create_env
from ap_rllib.helpers import sample_int, tune_int
from ray.tune.logger import pretty_print

def bursts_config(config, iteration):
    """Updates config to train with bursts, constant size."""
    config_new = deepcopy(config)

    pretrain_time = config['_train_steps'] // 2
    evaluation_time = config['_train_steps'] // 2
    burst_size = int(config['_burst_size'])

    # n_bursts = pretrain_time // (2 * burst_size)

    # print("Pretrain time: %d" % pretrain_time)
    # print("Evaluation time: %d" % evaluation_time)
    # print("Burst size", burst_size)
    # print("Number of bursts", n_bursts)
    # print("Total iterations (true)", n_bursts * burst_size * 2 + evaluation_time)

    train_policies = config_new['_train_policies']

    # pretraining stage
    if iteration < pretrain_time:
        if burst_size == 0:
            train_policies = ['player_1', 'player_2']
        elif burst_size < 0:
            raise ValueError("Wrong burst_size %s" % burst_size)
        else:
            current_burst = iteration // burst_size
            if current_burst % 2 == 0:
                train_policies = ['player_1']
            else:
                train_policies = ['player_2']
    else:
        train_policies = ['player_1']

    config_new['_train_policies'] = train_policies
    return config_new


def bursts_config_increase(config, iteration):
    """Updates config to train with bursts, exponentially increasing size."""
    config_new = deepcopy(config)

    train_time = config['_train_steps']
    evaluation_time = config['_eval_steps']
    exponent = config['_burst_exponent']

    if train_time + evaluation_time < iteration:
        print(f"Iteration {iteration} too high")

    info = {}
    # pretraining stage
    if iteration < train_time:
        bs_float, bs = 1.0, 1
        passed = 0
        while passed + 2 * bs < iteration + 1:
            passed += 2 * bs  # 2 agents in total
            bs_float = bs_float * exponent
            bs = round(bs_float)

        # last burst size is ours
        delta = iteration - passed
        first_stage = delta < bs

        currently_training = 'player_1' if first_stage else 'player_2'
        info['type'] = 'train'
        info['bs'] = bs
        info['bs_float'] = bs_float
        info['passed'] = passed
        info['delta'] = delta
    else:
        currently_training = 'player_1'
        info['type'] = 'eval'

    train_policies = []

    if '_all_policies' in config:
        all_policies = config['_all_policies']
    else:
        all_policies = ['player_1', 'player_2']

    for p in all_policies:
        if not p.startswith(currently_training):
            continue
        if p in config['_do_not_train_policies']:
            continue
        train_policies.append(p)

    config_new['_train_policies'] = train_policies
    config_new['_burst_info'] = info

    return config_new