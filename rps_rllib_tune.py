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

    policy_template = "learned%02d"

    agent_config = (PPOTFPolicy, obs_space, act_space, {
                    "model": {
                        "use_lstm": True,
                        "fcnet_hiddens": [config['fc_units'], config['fc_units']],
                        "lstm_cell_size": config['lstm_units'],
                    },
                    "framework": "tfe",
                })

    N_POLICIES = 2

    policies = {policy_template % i: agent_config for i in range(N_POLICIES)}
    policies_keys = list(sorted(policies.keys()))

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
    print(config)
    rl_config = build_trainer_config(restore_state=restore_state,
                              train_policies=config['train_policies'],
                              config=config)
    trainer = build_trainer(restore_state=None, config=rl_config)
    train(trainer, config['train_steps'], do_track=do_track)
    

node_sizes = [math.ceil(t) for t in np.logspace(0, 3, 10)]
batch_sizes = [128, 256, 512, 1024, 2048, 4096]
print(node_sizes)


# try changing learning rate
config = {'fc_units': tune.choice(node_sizes),
          'lstm_units': tune.choice(node_sizes),
          'train_batch_size': tune.choice(batch_sizes)}

config['train_steps'] = 100
config['train_policies'] = ['learned00']
config['num_workers'] = 22

print(config)

if __name__ == "__main__":
    ray_init()

    analysis = tune.run(
        train_one, 
        config=config, 
        verbose=1,
        num_samples=100,
        name="fixed_vs_learned",
    )
