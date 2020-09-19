import argparse
import json
import logging
import multiprocessing
import os
import pickle
import subprocess
import sys
import time
import uuid
from copy import deepcopy

import ray
import tensorflow as tf
from ray import tune
from ray.tune.logger import pretty_print
from sacred.observers import MongoObserver

# trials configuration
from config import CONFIGS, get_trainer
from helpers import dict_to_sacred, unlink_ignore_error

# parser for main()
parser = argparse.ArgumentParser(description='Train in YouShallNotPass')
parser.add_argument('--from_pickled_config', type=str, help='Trial to run (if None, run tune)', default=None,
                    required=False)
parser.add_argument('--tune', type=str, help='Run tune', default=None, required=False)
parser.add_argument('--tmp_dir', type=str, help='Temporary directory', default='/tmp', required=False)
parser.add_argument('--config_override', type=str, help='Config override json', default=None, required=False)


def ray_init(shutdown=True, tmp_dir='/tmp', **kwargs):
    """Initialize ray."""
    # number of CPUs on the machine
    num_cpus = multiprocessing.cpu_count()

    # restart ray / use existing session
    if shutdown:
        ray.shutdown()
    if not shutdown:
        kwargs['ignore_reinit_error'] = True

    # if address is not known, launch new instance
    if 'address' not in kwargs:
        # pretending we have more so that workers are never stuck
        # resources are limited by `tune_cpu` resources that we create
        kwargs['num_cpus'] = num_cpus * 2

        # `tune_cpu` resources are used to limit number of
        # concurrent trials
        kwargs['resources'] = {'tune_cpu': num_cpus}
        kwargs['temp_dir'] = tmp_dir

    # only showing errors, to prevent too many messages from coming
    kwargs['logging_level'] = logging.ERROR

    # launching ray
    return ray.init(log_to_driver=True, **kwargs)


def train_iteration_process(pickle_path):
    """Load config from pickled file, run and pickle the results."""

    # loading checkpoint/config from the file
    f = open(pickle_path, 'rb')
    checkpoint, config = pickle.load(f)

    # connecting to the existing ray session
    ray_init(shutdown=False, address=config['_redis_address'], tmp_dir=config['_tmp_dir'])

    # obtaining the trainer
    trainer = get_trainer(config)

    # restoring it
    if checkpoint:
        trainer.restore(checkpoint)
    iteration = trainer.iteration

    # restoring from a pre-defined checkpoint
    # doing it by copying weights only
    # so that iteration number is 0 instead of the saved one
    if '_checkpoint_restore' in config and iteration == 0:
        trainer_1 = get_trainer(config)
        trainer_1.restore(config['_checkpoint_restore'])
        trainer.set_weights(deepcopy(trainer_1.get_weights()))

    # doing one train interation and saving
    results = trainer.train()
    checkpoint = trainer.save()

    # formatting data
    results['checkpoint_rllib'] = checkpoint
    results['trainer_iteration'] = trainer.iteration
    del results['config']

    # stopping the trainer to free rollout worker processes
    trainer.stop()

    # saving data
    pickle.dump(results, open(pickle_path + '.ans.pkl', 'wb'))


def train_one_with_sacred(config, checkpoint_dir=None):
    """Train one trial with reporting to sacred."""
    do_track = True
    checkpoint = checkpoint_dir
    os.chdir(config['_base_dir'])

    if config['framework'] == 'tfe':
        tf.compat.v2.enable_v2_behavior()

    # setting a unique run id if necessary
    if 'run_uid' in config and config['run_uid'] == '_setme':
        config['run_uid'] = str(uuid.uuid1())

    # creating a sacred experiment
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

        def train_iteration(checkpoint, config):
            """One training iteration with subprocess."""
            # saving configuration to a pickle file
            pickle_path = config['_tmp_dir'] + '/' + str(uuid.uuid1()) + '.pkl'
            pickle_path_ans = pickle_path + '.ans.pkl'
            pickle.dump([checkpoint, config], open(pickle_path, 'wb'))

            # running subprocess or running inline
            # overhead doesn't seem significant!	
            if config['_run_inline']:
                train_iteration_process(pickle_path)
            elif config['_log_error']:
                subprocess.run(
                    "python %s --from_pickled_config %s 2>&1 > %s" % (config['_main_filename'], pickle_path,
                                                                      pickle_path + '.err'),
                    shell=True)
            else:
                subprocess.run(
                    "python %s --from_pickled_config %s" % (config['_main_filename'], pickle_path),
                    shell=True)

            # obtaining results
            try:
                results = pickle.load(open(pickle_path_ans, 'rb'))
                unlink_ignore_error(pickle_path)
                unlink_ignore_error(pickle_path_ans)
                unlink_ignore_error(pickle_path + '.err')
            except:
                time.sleep(5)
                print(open(pickle_path + '.err', 'r').read())
                raise Exception("Train subprocess has failed, error %s" % (pickle_path + '.err'))
            return results

        iteration = 0
        # running iterations	
        while True:
            # doing edits in the config, bursts, for example
            config_updated = config
            if config['_update_config']:
                config_updated = config['_update_config'](config, iteration)
            results = train_iteration(checkpoint, config_updated)
            checkpoint = results['checkpoint_rllib']
            iteration = results['trainer_iteration']
            print("Iteration %d done" % iteration)

            # reporting
            dict_to_sacred(ex, results, iteration)

            if do_track:
                tune.report(**results)
            else:
                print(pretty_print(results))

            # stopping at the end
            if iteration > config['_train_steps']:
                return

    return ex.run()


def run_tune(config_name=None, config_override=None, tmp_dir=None):
    """Call tune."""
    assert config_name in CONFIGS, "Wrong config %s" % str(list(CONFIGS.keys()))
    config = CONFIGS[config_name]
    cluster_info = ray_init(tmp_dir=tmp_dir)
    tf.keras.backend.set_floatx('float32')

    # run metadata
    config['_main_filename'] = os.path.realpath(__file__)
    config['_redis_address'] = cluster_info['redis_address']
    config['_base_dir'] = os.path.dirname(os.path.realpath(__file__))
    config['_tmp_dir'] = tmp_dir

    # changing config entries from command line
    if config_override:
        config_override = json.loads(config_override)
        for k, v in config_override.items():
            config[k] = v

    # running tune
    tune.run(
        train_one_with_sacred,
        config=config,
        verbose=True,
        queue_trials=True,
        **config['_call'],
    )


# main script: command-line interface
if __name__ == '__main__':
    args = parser.parse_args()

    # this option runs 1 training iteration
    if args.from_pickled_config:
        train_iteration_process(pickle_path=args.from_pickled_config)

    # this option runs tune trials
    elif args.tune:
        run_tune(config_name=args.tune, config_override=args.config_override, tmp_dir=args.tmp_dir)
    else:
        parser.print_help()

    sys.exit(0)
