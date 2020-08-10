#!/usr/bin/env python
# coding: utf-8

import uuid
import subprocess
import sys
import argparse
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

def ray_init(num_cpus=60, shutdown=True):
    """Initialize ray."""
    if shutdown:
        ray.shutdown()
    kwargs = {}
    if not shutdown:
        kwargs['ignore_reinit_error'] = True
    return ray.init(num_cpus=num_cpus * 2, log_to_driver=False,
            temp_dir='/scratch/sergei/tmp', resources={'tune_cpu': num_cpus,}, **kwargs)


def build_trainer_config(train_policies, config, num_workers=4, use_gpu=True, num_workers_tf=32, env_config=env_config, load_normal=False):
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
                    "observation_filter": "MeanStdFilter",
                })
        
        if agent_id == 1:
            return agent_config_pretrained if load_normal else agent_config_from_scratch
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
        "use_gae": True,
        "num_gpus": 4 if use_gpu else 0,
        "batch_mode": "complete_episodes",
        "num_workers": num_workers,
        "train_batch_size": int(config['train_batch_size']),
        "rollout_fragment_length": int(config.get('rollout_fragment_length', 200)),
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

def train_iteration_process(pickle_path, ray_init=True):
    print("Load pickle")
    checkpoint, config = pickle.load(open(pickle_path, 'rb'))
    print("Ray init")
    if ray_init:
        ray.init(address=config['redis_address'])
    #ray.init(num_cpus=5, include_dashboard=False)
    print("Build config")
    rl_config = build_trainer_config(train_policies=config['train_policies'],
                              config=config)
    print("Create trainer")
    trainer = PPOTrainer(config=rl_config)
    print("Restore")
    if checkpoint:
        trainer.restore(checkpoint)
    print("Train")
    results = trainer.train()
    print("Save")
    checkpoint = trainer.save()
    results['checkpoint_rllib'] = checkpoint
    results['trainer_iteration'] = trainer.iteration
    del results['config']
    print("Dump")
    trainer.stop()
    pickle.dump(results, open(pickle_path + '.ans.pkl', 'wb'))
    #ray.shutdown()
    print("Done")

def train_one(config, checkpoint=None, do_track=True):
    print("Building trainer config...")
    print("CONFIG", config)
    print("CHECKPOINT", checkpoint)
    if not isinstance(checkpoint, str):
        checkpoint = None
    restore_state = None
    
    print("Starting iterations...")
   
    def train_iteration(checkpoint, config):
        pickle_path = '/tmp/' + str(uuid.uuid1()) + '.pkl'
        pickle_path_ans = pickle_path + '.ans.pkl'
        pickle.dump([checkpoint, config], open(pickle_path, 'wb'))
        # overhead doesn't seem significant!
        subprocess.run("python %s --one_trial %s 2>&1 > %s" % (config['main_filename'], pickle_path, pickle_path + '.err'), shell=True)
        #train_iteration_process(pickle_path, ray_init=False)
        os.unlink(pickle_path)
        try:
            results = pickle.load(open(pickle_path_ans, 'rb'))
            os.unlink(pickle_path_ans)
        except:
            raise Exception("Train subprocess has failed, error %s" % (pickle_path + '.err'))
        return results

    print("CHECKPOINT", str(checkpoint), config)

    while True:
        results = train_iteration(checkpoint, config)
        checkpoint = results['checkpoint_rllib']
        iteration = results['trainer_iteration']
        print("Iteration %d done" % iteration)

        if do_track:
            tune.report(**results)
        else:
            print(pretty_print(results))
            print("Checkpoint", checkpoint)

        if iteration > config['train_steps']:
            return

def get_config_coarse():
    # try changing learning rate
    config = {}

    config['train_batch_size'] = tune.loguniform(2048, 320000, 2)
    config['lr'] = tune.loguniform(1e-5, 1e-2, 10)
    config['sgd_minibatch_size'] = tune.loguniform(512, 65536, 2)
    config['num_sgd_iter'] = tune.uniform(1, 30)
    config['train_steps'] = 99999999
    config['rollout_fragment_length'] = tune.loguniform(200, 5000, 2)

    # ['humanoid_blocker', 'humanoid'],
    config['train_policies'] = ['player_1']
    return config

def get_config_fine():
    # try changing learning rate
    config = {}

    config['train_batch_size'] = tune.loguniform(2048, 150000, 2)
    config['lr'] = tune.loguniform(1e-5, 1e-3, 10)
    config['sgd_minibatch_size'] = tune.loguniform(1000, 65536, 2)
    config['num_sgd_iter'] = tune.uniform(3, 30)
    config['train_steps'] = 99999999
    config['rollout_fragment_length'] = tune.loguniform(2000, 5000, 2)

    # ['humanoid_blocker', 'humanoid'],
    config['train_policies'] = ['player_1']
    return config

def get_config_small():
    # try changing learning rate
    config = {}

    config['train_batch_size'] = 128
    config['lr'] = 1e-4
    config['sgd_minibatch_size'] = 128
    config['num_sgd_iter'] = 2
    config['train_steps'] = 99999999
    config['rollout_fragment_length'] = 200

    # ['humanoid_blocker', 'humanoid'],
    config['train_policies'] = ['player_1']
    return config


def main(_run=None):
    config = get_config_fine()
    cluster_info = ray_init()
    print(cluster_info)
    custom_scheduler = ASHAScheduler(
        metric='policy_reward_mean/player_1',
        mode="max",
        grace_period=1000000,
        reduction_factor=2,
        max_t=50000000,
        time_attr='timesteps_total',
    )
    tf.keras.backend.set_floatx('float32')

    config['main_filename'] = sys.argv[0]
    config['redis_address'] = cluster_info['redis_address']

    #config = {}
    #config['train_batch_size'] = 320000
    #config['lr'] = 3e-4
    #config['num_sgd_iter'] = 20
    #config['sgd_minibatch_size'] = 32768
    #config['train_policies'] = ['player_1']
    #config['train_steps'] = 10000

    #train_one(config=config, do_track=False)
    #return

    analysis = tune.run(
            train_one, 
            config=config, 
            verbose=True,
            name="adversarial_tune_fine",
            num_samples=300,
            checkpoint_freq=0, # checkpoints done by the function itself
            #scheduler=custom_scheduler,
            resources_per_trial={"custom_resources": {"tune_cpu": 4}},
            queue_trials=True,
            #resume=True,
            stop={'timesteps_total': 50000000} # 30 million time-steps
        )


parser = argparse.ArgumentParser(description='Train in YouShallNotPass')
parser.add_argument('--one_trial', type=str, help='Trial to run (if None, run tune)', default=None, required=False)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.one_trial is None:
        main()
    else:
        train_iteration_process(args.one_trial)
        sys.exit(0)
