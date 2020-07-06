#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from rps_rllib import RPSNoise
import numpy as np
import ray
from ray.rllib import agents
from tqdm.notebook import tqdm
import random
from ray.rllib.examples.env.rock_paper_scissors import RockPaperScissors
from ray.rllib.policy.policy import Policy
from gym.spaces import Discrete, Box
from ray.rllib.agents.ppo import PPOTrainer
from functools import partial
from ray.tune.registry import register_env, _global_registry, ENV_CREATOR
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

import ray
from ray import tune
from ray.tune import track

import math


def ray_init():
    ray.shutdown()
    return ray.init(ignore_reinit_error=True, include_webui=True,
                    temp_dir='/scratch/sergei/tmp') # Skip or set to ignore if already called

ray_init()


env_config = {}
env_cls = RockPaperScissors


def build_trainer_config(restore_state=None, train_policies=None, config=None):
    """Build configuration for 1 run."""
    obs_space = env_cls(env_config).observation_space
    act_space = env_cls(env_config).action_space

    agent_config = (PPOTFPolicy, obs_space, act_space, {
                    "model": {
                        "use_lstm": True,
                        "fcnet_hiddens": [config['fc_units'], config['fc_units']],
                        "lstm_cell_size": config['lstm_units'],
                    },
                    "framework": "tfe",
                })

    # N_POLICIES = 2
    policies_keys = ['victim', 'adversary']

    #policies = {policy_template % i: agent_config for i in range(N_POLICIES)}
    policies = {name: agent_config for name in policies_keys}

    def select_policy(agent_id):
        assert agent_id in ["player1", "player2"]
        agent_ids = ["player1", "player2"]
        
        # selecting the corresponding policy (only for 2 policies)
        return policies_keys[agent_ids.index(agent_id)]

        # randomly choosing an opponent
        # return np.random.choice(list(policies.keys()))
    
    if train_policies is None:
        train_policies = list(policies.keys())
        
    for k in train_policies:
        assert k in policies.keys()

    config = {
        "env": env_cls,
    #    "gamma": 0.9,
      "num_workers": 0,
    #  "num_envs_per_worker": 10,
    #   "rollout_fragment_length": 10,
       "train_batch_size": config['train_batch_size'],
        "multiagent": {
            "policies_to_train": train_policies,
            "policies": policies,
            "policy_mapping_fn": select_policy,
        },
        "framework": "tfe",
        #"train_batch_size": 512
        #"num_cpus_per_worker": 2
    }
    return config


def build_trainer(restore_state=None, train_policies=None, config=None):
    """Create a RPS trainer for 2 agents, restore state, and only train specific policies."""
    
    print("Using config")
    print(config)
    cls = PPOTrainer
    trainer = cls(config=config)
    env = trainer.workers.local_worker().env
    if restore_state is not None:
        trainer.restore_from_object(restore_state)
    return trainer

def train(trainer, stop_iters, do_track=True):
    """Train the agents and return the state of the trainer."""
    for _ in range(stop_iters):
        results = trainer.train()
        print(pretty_print(results))
        if do_track:
            track.log(**results)
    o = trainer.save_to_object()
    return o


def train_one(config, restore_state=None, do_track=True):
    """Train with one config."""

    def train_call(policies, state, iters):
        rl_config = build_trainer_config(restore_state=state,
                                  train_policies=policies,
                                  config=config)
        trainer = build_trainer(restore_state=state, config=rl_config)
        state = train(trainer, iters, do_track=do_track)
        return state
    
    pretrain_time = config['train_steps']
    evaluation_time = config['train_steps']
    burst_size = config['burst_size']
    
    n_bursts = pretrain_time // (2 * burst_size)
    
    print("Pretrain time: %d" % pretrain_time)
    print("Evaluation time: %d" % evaluation_time)
    print("Burst size", burst_size)
    print("Number of bursts", n_bursts)
    print("Total iterations (true)", n_bursts * burst_size * 2 + evaluation_time)
    
    state = None
    
    if burst_size == 0:
        state = train_call(['victim', 'adversary'], state, pretrain_time)
    else:
        for i in range(n_bursts):
            state = train_call(['victim'], state, burst_size)
            state = train_call(['adversary'], state, burst_size)
        
    state = train_call(['adversary'], state, evaluation_time)
    
    return state
    
    

burst_sizes = list(np.arange(35))

# best hypers from rps_rllib_tune.py and rps_rllib-analysis.ipynb
config = {
    'fc_units':                         100,
    'lstm_units':                       22,
    'num_workers':                      10,
    'train_batch_size':                 4096,
    'train_steps':                      40,
}

config['burst_size'] = tune.grid_search(burst_sizes)

print(config)

if __name__ == "__main__":
    ray_init()

    analysis = tune.run(
        train_one, 
        config=config, 
        verbose=1,
        #num_samples=100,
        name="bursts",
        num_samples=10,
    )