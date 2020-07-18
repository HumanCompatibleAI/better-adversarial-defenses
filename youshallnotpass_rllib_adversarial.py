#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import numpy as np
import ray
from ray.rllib import agents
from tqdm.notebook import tqdm
import random
from ray.rllib.policy.policy import Policy
from gym.spaces import Discrete, Box
from ray.rllib.agents.ppo import PPOTrainer
from functools import partial
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.models import ModelCatalog
import json, pickle

import ray
from ray import tune
from ray.tune import track

import math
import gym

from gym_compete_to_rllib import created_envs, env_name, create_env, env_name_rllib

import os
os.environ['DISPLAY'] = ':0'
import codecs

def ray_init():
    ray.shutdown()
    return ray.init(ignore_reinit_error=True,
                    temp_dir='/scratch/sergei/tmp') # Skip or set to ignore if already called

ray_init()


# In[ ]:


tf.keras.backend.set_floatx('float32')


# In[ ]:


env_cls = create_env
env_config = {'with_video': False}#True}

def build_trainer_config(restore_state=None, train_policies=None, config=None, num_workers=8, env_config=env_config):
    """Build configuration for 1 run."""
    obs_space = env_cls(env_config).observation_space
    act_space = env_cls(env_config).action_space

    policy_template = "player_%d"

    def get_agent_config(agent_id):
        agent_config_pretrained = (PPOTFPolicy, obs_space, act_space, {
            'model': {
                        "custom_model": "GymCompetePretrainedModel",
                        "custom_model_config": {
                            "agent_id": agent_id - 1,
                            "env_name": env_name,
                            "model_config": {},
                            "name": "model_%s" % (agent_id - 1)
                        },           
                        
                    },
            
            "framework": "tfe",
        })
        
        agent_config_from_scratch = (PPOTFPolicy, obs_space, act_space, {
                    "model": {
                        "use_lstm": False,
                        "fcnet_hiddens": [64, 64],
                        #"custom_action_dist": "DiagGaussian",
                    },
                    "framework": "tfe",
                })
        
        if agent_id == 1:
            return agent_config_from_scratch
        elif agent_id == 2:
            return agent_config_pretrained
        else:
            raise KeyError("Wrong agent id %s" % agent_id)

    N_POLICIES = 2

    policies = {policy_template % i: get_agent_config(i) for i in range(1, 1  + N_POLICIES)}
    policies_keys = list(sorted(policies.keys()))

    def select_policy(agent_id):
        assert agent_id in ["player_1", "player_2"]
        agent_ids = ["player_1", "player_2"]
        
        # selecting the corresponding policy (only for 2 policies)
        return policies_keys[agent_ids.index(agent_id)]

        # randomly choosing an opponent
        # return np.random.choice(list(policies.keys()))
    
    if train_policies is None:
        train_policies = list(policies.keys())
        
    for k in train_policies:
        assert k in policies.keys()

    config = {
        "env": env_name_rllib,
        "env_config": env_config,
    #    "gamma": 0.9,
      "num_workers": num_workers,
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

def train_iteration(trainer, stop_iters, do_track=True):
    """Train the agents and return the state of the trainer."""
    for _ in range(stop_iters):
        results = trainer.train()
        print(pretty_print(results))
        if do_track:
            track.log(**results)
    o = trainer.save_to_object()
    return o

trainer = None

def train_one(config, checkpoint=None):
    start = 0
    if checkpoint:
        with open(checkpoint) as f:
            state = json.loads(f.read())
            start = state["step"] + 1
            
    restore_state = None
    do_track = True
    rl_config = build_trainer_config(restore_state=restore_state,
                              train_policies=config['train_policies'],
                              config=config)
    global trainer
    trainer = build_trainer(restore_state=None, config=rl_config)

    for step in range(start, config['train_steps']):
        results = trainer.train()
        print(pretty_print(results))
        if do_track:
            track.log(**results)


        if step % 10 == 0:
            # Obtain a checkpoint directory
            checkpoint_dir = tune.make_checkpoint_dir(step=step)
            path = os.path.join(checkpoint_dir, "checkpoint")
            with open(path, "w") as f:
                w = trainer.get_weights()
                wp = pickle.dumps(w)
                wps = codecs.encode(wp, 'base64').decode()

                f.write(json.dumps({"step": start, "weights": wps}))
            tune.save_checkpoint(path)

        
# try changing learning rate
config = {'train_batch_size': 4096}

config['train_steps'] = 10000

# ['humanoid_blocker', 'humanoid'],
config['train_policies'] = ['player_1']
#config['num_workers'] = 8


# In[ ]:


#train_one(config, do_track=False)
if __name__ == '__main__':
    analysis = tune.run(
            train_one, 
            config=config, 
            verbose=1,
            #num_samples=100,
            name="adversarial",
            num_samples=10,
            checkpoint_freq=10 
        )


    # In[ ]:


    [x.close() for x in created_envs]
