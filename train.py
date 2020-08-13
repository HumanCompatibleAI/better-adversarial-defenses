#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pickle
import subprocess
import sys
import uuid

import ray
import tensorflow as tf
from ray import tune
from ray.tune.logger import pretty_print
from sacred.observers import MongoObserver

# trials configuration
from config import CONFIGS, TRAINERS, get_agent_config
import tensorflow as tf
import json

tf.compat.v2.enable_v2_behavior()

# parser for main()
parser = argparse.ArgumentParser(description='Train in YouShallNotPass')
parser.add_argument('--from_pickled_config', type=str, help='Trial to run (if None, run tune)', default=None,
                    required=False)
parser.add_argument('--tune', type=str, help='Run tune', default=None, required=False)
parser.add_argument('--config_override', type=str, help='Config override json', default=None, required=False)



def ray_init(num_cpus=28, shutdown=True):
    """Initialize ray."""
    if shutdown:
        ray.shutdown()
    kwargs = {}
    if not shutdown:
        kwargs['ignore_reinit_error'] = True
    return ray.init(num_cpus=num_cpus * 2, num_gpus=1, # log_to_driver=False,
                    temp_dir='/scratch/sergei/tmp', resources={'tune_cpu': num_cpus, }, **kwargs)


def build_trainer_config(config):
    """Build configuration for 1 run."""
    # determining environment parameters
    env_fcn = config['_env_fcn']
    env = env_fcn(config['_env'])
    obs_space, act_space, n_policies = env.observation_space, env.action_space, env.n_policies
    env.close()

    # creating policies
    policy_template = "player_%d"

    policies = {policy_template % i: get_agent_config(agent_id=i, which=config['_policies'][i],
                                                      config=config,
                                                      obs_space=obs_space, act_space=act_space)
                for i in range(1, 1 + n_policies)}
    policies_keys = list(sorted(policies.keys()))

    def select_policy(agent_id):
        """Select policy at execution."""
        agent_ids = ["player_1", "player_2"]
        assert agent_id in agent_ids

        # selecting the corresponding policy (only for 2 policies)
        return policies_keys[agent_ids.index(agent_id)]

    for k in config['_train_policies']:
        assert k in policies.keys()

    rl_config = {
        "env": config['_env_name_rllib'],
        "env_config": config['_env'],
        "multiagent": {
            "policies_to_train": config['_train_policies'],
            "policies": policies,
            "policy_mapping_fn": select_policy,
        },
        'tf_session_args': {'intra_op_parallelism_threads': config['_num_workers_tf'],
                            'inter_op_parallelism_threads': config['_num_workers_tf'],
                            'gpu_options': {'allow_growth': True},
                            'log_device_placement': True,
                            'device_count': {'CPU': config['_num_workers_tf']},
                            'allow_soft_placement': True
                            },
        "local_tf_session_args": {
            "intra_op_parallelism_threads": config['_num_workers_tf'],
            "inter_op_parallelism_threads": config['_num_workers_tf'],
        },
    }

    # filling in the rest of variables
    for k, v in config.items():
        if k.startswith('_'): continue
        rl_config[k] = v

    print("Config:", pretty_print(rl_config))

    return rl_config


def train_iteration_process(pickle_path, ray_init=True):
    """Load config from pickled file, run and pickle the results."""
    print(pickle_path, ray_init)
    f = open(pickle_path, 'rb')
    checkpoint, config = pickle.load(f)
    if ray_init:
        ray.init(address=config['_redis_address'])
    rl_config = build_trainer_config(config=config)
    trainer = TRAINERS[config['_trainer']](config=rl_config)
    if checkpoint:
        trainer.restore(checkpoint)
    results = trainer.train()
    checkpoint = trainer.save()
    results['checkpoint_rllib'] = checkpoint
    results['trainer_iteration'] = trainer.iteration
    del results['config']
    trainer.stop()
    pickle.dump(results, open(pickle_path + '.ans.pkl', 'wb'))


def dict_to_sacred(ex, d, iteration, prefix=''):
    """Log a dictionary to sacred."""
    for k, v in d.items():
        if isinstance(v, dict):
            dict_to_sacred(ex, v, iteration, prefix=prefix + k + '/')
        elif isinstance(v, float) or isinstance(v, int):
            ex.log_scalar(prefix + k, v, iteration)

def train_one_with_sacred(config, checkpoint=None, do_track=True):
    #print(config['_base_dir'])
    #print(os.getcwd())
    os.chdir(config['_base_dir'])

    tf.compat.v2.enable_v2_behavior()

    # https://github.com/IDSIA/sacred/issues/492
    from sacred import Experiment, SETTINGS
    SETTINGS.CONFIG.READ_ONLY_CONFIG = False

    ex = Experiment(config['_call']['name'], base_dir=config['_base_dir'])
    ex.observers.append(MongoObserver(db_name='chai'))
    ex.add_source_file('config.py')

    ex.add_config(config=config, checkpoint=checkpoint, do_track=do_track, **config)

    @ex.main
    def train_one(config, checkpoint=None, do_track=True):
        """One trial with subprocesses for each iteration."""
        if not isinstance(checkpoint, str):
            checkpoint = None

        print("Starting iterations...")

        global trainer
        trainer = None
        
        if '_checkpoint_restore' in config:
            checkpoint = config['_checkpoint_restore']

        def train_iteration_inline(checkpoint, config):
            """Load config from pickled file, run and pickle the results."""
            global trainer

            if trainer is None:
                rl_config = build_trainer_config(config=config)
                trainer = TRAINERS[config['_trainer']](config=rl_config)
                if checkpoint:
                    trainer.restore(checkpoint)
            results = trainer.train()
            checkpoint = trainer.save()
            results['checkpoint_rllib'] = checkpoint
            results['trainer_iteration'] = trainer.iteration
            return results

        def train_iteration(checkpoint, config):
            """One training iteration with subprocess."""
            pickle_path = '/tmp/' + str(uuid.uuid1()) + '.pkl'
            pickle_path_ans = pickle_path + '.ans.pkl'
            pickle.dump([checkpoint, config], open(pickle_path, 'wb'))
            # overhead doesn't seem significant!
            subprocess.run(
                "python %s --from_pickled_config %s 2>&1 > %s" % (config['_main_filename'], pickle_path,
                                                                  pickle_path + '.err'),
                shell=True)
            # train_iteration_process(pickle_path, ray_init=False)
            try:
                results = pickle.load(open(pickle_path_ans, 'rb'))
                os.unlink(pickle_path)
                os.unlink(pickle_path_ans)
            except:
                print(open(pickle_path + '.err', 'r').read())
                raise Exception("Train subprocess has failed, error %s" % (pickle_path + '.err'))
            return results

        iteration = 0
        # running iterations
        while True:
            config_updated = config
            if config['_update_config']:
                config_updated = config['_update_config'](config, iteration)
            if config['_run_inline']:
                results = train_iteration_inline(checkpoint, config_updated)
            else:
                results = train_iteration(checkpoint, config_updated)
            checkpoint = results['checkpoint_rllib']
            iteration = results['trainer_iteration']
            print("Iteration %d done" % iteration)
            dict_to_sacred(ex, results, iteration)

            if do_track:
                tune.report(**results)
            else:
                print(pretty_print(results))

            if iteration > config['_train_steps']:
                return

    return ex.run()


def run_tune(config_name=None, config_override=None):
    """Call tune."""
    assert config_name in CONFIGS, "Wrong config %s" % str(list(CONFIGS.keys()))
    config = CONFIGS[config_name]
    cluster_info = ray_init()
    tf.keras.backend.set_floatx('float32')

    config['_main_filename'] = os.path.realpath(__file__)
    config['_redis_address'] = cluster_info['redis_address']
    config['_base_dir'] = os.path.dirname(os.path.realpath(__file__))

    if config_override:
        config_override = json.loads(config_override)
        for k, v in config_override.items():
            config[k] = v

    analysis = tune.run(
        train_one_with_sacred,
        config=config,
        verbose=True,
        queue_trials=True,
        **config['_call'],
    )


if __name__ == '__main__':
    args = parser.parse_args()
    if args.from_pickled_config:
        train_iteration_process(pickle_path=args.from_pickled_config)
    elif args.tune:
        run_tune(config_name=args.tune, config_override=args.config_override)
    else:
        parser.print_help()

    sys.exit(0)
