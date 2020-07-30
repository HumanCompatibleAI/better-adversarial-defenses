#!/usr/bin/env python
# coding: utf-8

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

import math
import gym

from gym_compete_to_rllib import created_envs, env_name, create_env, env_name_rllib

import os
#os.environ['DISPLAY'] = ':0'
import codecs
import time

from ray.tune.schedulers import ASHAScheduler

from sacred import Experiment
from sacred.observers import MongoObserver

#ex = Experiment("youshallnotpass_learned_adversary_vs_zoo", interactive=True)
#ex.observers.append(MongoObserver(url='127.0.0.1:27017',
#                                      db_name='better_adversarial_defenses'))

env_cls = create_env
env_config = {'with_video': False}

#@ex.capture
def log_dict(d, prefix='', counter=0, _run=None):
    """Ray dictionary results to sacred."""
    for k, v in d.items():
        if isinstance(v, int) or isinstance(v, float):
            _run.log_scalar(k, v, counter)
        elif isinstance(v, dict):
            log_dict(d=d, prefix=k + '.', counter=counter)

def ray_init(num_cpus=60):
    """Initialize ray."""
    ray.shutdown()
    return ray.init(num_cpus=num_cpus, # ignore_reinit_error=True
            temp_dir='/scratch/sergei/tmp', resources={'tune_cpu': num_cpus,})


def build_trainer_config(train_policies, config, num_workers=10, num_workers_tf=32, env_config=env_config):
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
                        "fcnet_activation": "tanh",
                        "free_log_std": True,
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
        "use_gae": False,
        "num_gpus": 1,
        "batch_mode": "complete_episodes",
        "observation_filter": "MeanStdFilter",
        "num_workers": num_workers,
        "train_batch_size": int(config['train_batch_size']),
        "multiagent": {
            "policies_to_train": train_policies,
            "policies": policies,
            "policy_mapping_fn": select_policy,
        },
        "framework": "tfe",
        "lr": config.get('lr', 1e-4),
        "vf_loss_coeff": 0.5,
        "gamma": 0.995,
        "sgd_minibatch_size": int(config.get("sgd_minibatch_size", 128)),
        "num_sgd_iter": int(config.get("num_sgd_iter", 30)),
        
        'tf_session_args': {'intra_op_parallelism_threads': num_workers_tf,
          'inter_op_parallelism_threads': num_workers_tf,
          'gpu_options': {'allow_growth': True},
          'log_device_placement': True,
          'device_count': {'CPU': num_workers_tf},
          'allow_soft_placement': True
        },
        
        "local_tf_session_args": {
            "intra_op_parallelism_threads": num_workers_tf,
            "inter_op_parallelism_threads": num_workers_tf,
        },
        "kl_coeff": 1.0,
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


def train_one(config, checkpoint=None, _run=None, do_track=True):
    start = 0
    w = None
    if checkpoint:
        with open(checkpoint, 'r') as f:
            state = json.loads(f.read())
            start = state["step"] + 1
            w = codecs.decode(state['weights'].encode(), 'base64')
            
    print("Building trainer config...")
    restore_state = None
    rl_config = build_trainer_config(train_policies=config['train_policies'],
                              config=config)
    
    print("Building trainer...")
    trainer = build_trainer(restore_state=None, config=rl_config)
    # trainer.restore(os.path.dirname(checkpoint) + "/")
    if w is not None:
        trainer.set_weights(pickle.loads(w))

    if checkpoint is None:
        checkpoint = trainer.logdir
    
    print("Starting iterations...")
    
    for step in range(start, config['train_steps']):
        results = trainer.train()
        print("Iteration %d done" % step)
        if do_track:
            tune.report(**results)
        else:
            print(pretty_print(results))
        if _run is not None:
            log_dict(d=results, counter=step)


        if step % 10 == 0 or step == start:
            # Obtain a checkpoint directory
            if do_track:
                checkpoint_dir = tune.make_checkpoint_dir(step=step)
            else:
                checkpoint_dir = checkpoint + "_new"
            path = os.path.join(checkpoint_dir, str(step), "checkpoint")
            os.makedirs(os.path.join(checkpoint_dir, str(step)), exist_ok=True)
            with open(path, "w") as f:
                w = trainer.get_weights()
                wp = pickle.dumps(w)
                wps = codecs.encode(wp, 'base64').decode()

                f.write(json.dumps({"step": step, "weights": wps}))
            if do_track:
                tune.save_checkpoint(path)
            print("Checkpoint %d done" % step)


def get_config():
    # try changing learning rate
    config = {}

    config['train_batch_size'] = tune.loguniform(2048, 65536, 2)
    config['lr'] = tune.loguniform(1e-5, 1e-2, 10)
    config['sgd_minibatch_size'] = tune.loguniform(512, 65536, 2)
    config['num_sgd_iter'] = tune.uniform(1, 30)
    config['train_steps'] = 10000

    # ['humanoid_blocker', 'humanoid'],
    config['train_policies'] = ['player_1']
    return config

def main(_run=None):
    config = get_config()
    ray_init()
    custom_scheduler = ASHAScheduler(
        metric='policy_reward_mean/player_1',
        mode="max",
        grace_period=10 * 3600,
        max_t=5 * 24 * 3600,
        time_attr='time_since_restore',
    )
    tf.keras.backend.set_floatx('float32')


    config = {}
    config['train_batch_size'] = 320000
    config['lr'] = 3e-4
    config['num_sgd_iter'] = 20
    config['sgd_minibatch_size'] = 32768
    config['train_policies'] = ['player_1']
    config['train_steps'] = 10000

    train_one(config=config, do_track=False)
    return

    analysis = tune.run(
            train_one, 
            config=config, 
            verbose=True,
            name="adversarial_tune",
            num_samples=1,
            checkpoint_freq=10,
            scheduler=custom_scheduler,
            resources_per_trial={"custom_resources": {"tune_cpu": 9}},
            queue_trials=True,
            resume=True,
        )

if __name__ == '__main__':
    main()
