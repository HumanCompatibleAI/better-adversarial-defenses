import tensorflow as tf
import numpy as np
import ray
from ray.rllib import agents
from tqdm.notebook import tqdm
import random
from ray.rllib.policy.policy import Policy
from gym.spaces import Discrete, Box
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
import ray
from ray import tune
import math
import os
import time
from dummy_multiagent_env import create_env, env_name

tf.compat.v1.enable_eager_execution()

env_cls = create_env
env_config = {}


def ray_init(num_cpus=None, shutdown=True):
    """Initialize ray."""
    if shutdown:
        ray.shutdown()
    kwargs = {}
    if not shutdown:
        kwargs['ignore_reinit_error'] = True
    return ray.init(num_cpus=num_cpus, **kwargs)


def get_trainer_config(env_config, num_workers=1, num_workers_tf=32):
    """Build configuration for 1 run."""
    env = env_cls(env_config)
    obs_space = env.observation_space
    act_space = env.action_space
    del env

    policy_template = "player_%d"

    def get_agent_config(agent_id):
        agent_config = (PPOTFPolicy, obs_space, act_space, {
            "model": {
                # "use_lstm": False,
                # "fcnet_hiddens": [64, 64],
                # "custom_action_dist": "DiagGaussian",
                # "fcnet_activation": "tanh",
                # "free_log_std": True,
            },
            "framework": "tfe",
            # "observation_filter": "MeanStdFilter",
        })
        return agent_config

    N_POLICIES = 2

    policies = {policy_template % i: get_agent_config(i) for i in range(1, 1 + N_POLICIES)}
    policies_keys = list(sorted(policies.keys()))

    def select_policy(agent_id):
        agent_ids = ["player_%d" % i for i in range(1, 1 + N_POLICIES)]
        assert agent_id in agent_ids

        # selecting the corresponding policy
        return policies_keys[agent_ids.index(agent_id)]

    for k in train_policies:
        assert k in policies.keys()

    config = {
        "env": env_name,
        "env_config": env_config,
        # "use_gae": False,
        # "num_gpus": 1 if use_gpu else 0,
        # "batch_mode": "complete_episodes",
        "num_workers": num_workers,
        "train_batch_size": 128,
        "multiagent": {
            "policies_to_train": train_policies,
            "policies": policies,
            "policy_mapping_fn": select_policy,
        },
        "framework": "tfe",
        "lr": 1e-3,
        "vf_loss_coeff": 0.5,
        "gamma": 0.995,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 5,

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


def main():
    ray_init()
    tf.keras.backend.set_floatx('float32')

    config = get_trainer_config(train_policies=['player_1', 'player_2'],
                                config=config, env_config=env_config)
    config['train_batch_size'] = 32768
    config['lr'] = 3e-4
    config['num_sgd_iter'] = 10
    config['sgd_minibatch_size'] = 8192
    config['train_steps'] = 10000

    print(config)

    analysis = tune.run(
        "PPO",  # train_one,
        config=config,
        verbose=True,
        name="dummy_run",
        num_samples=1,
        checkpoint_freq=10,
        queue_trials=True,
    )


if __name__ == '__main__':
    main()
