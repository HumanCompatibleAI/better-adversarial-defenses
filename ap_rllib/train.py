import argparse
import json
import os
import pickle
import subprocess
import sys
import tensorflow as tf
import time
import uuid
from copy import deepcopy
from ray.tune.logger import pretty_print
import logging
from sacred.observers import MongoObserver

from ap_rllib.config import get_trainer, get_config_by_name, select_config, get_config_names
from ap_rllib.helpers import dict_to_sacred, unlink_ignore_error, ray_init
from ap_rllib_experiment_analysis.analysis_helpers import get_df_from_logdir
from ray import tune

# parser for main()
parser = argparse.ArgumentParser(description='Train in YouShallNotPass')
parser.add_argument('--from_pickled_config', type=str, help='Trial to run (if None, run tune)', default=None,
                    required=False)
parser.add_argument('--tune', type=str, help='Run tune', default=None, required=False, choices=get_config_names())
parser.add_argument('--tmp_dir', type=str, help='Temporary directory', default='/tmp', required=False)
parser.add_argument('--config_override', type=str, help='Config override json', default=None, required=False)
parser.add_argument('--verbose', action='store_true', required=False)
parser.add_argument('--resume', action='store_true', required=False, help="Resume all trials from the checkpoint.")
parser.add_argument('--show_config', action='store_true', required=False, help="Only show config (no train)")

logger = logging.getLogger('train_script')


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
        if '_restore_only' in config and '_foreign_config' in config:
            foreign_config = get_config_by_name(config['_foreign_config'])
            foreign_config['_verbose'] = False
            trainer_1 = get_trainer(foreign_config)
            trainer_1.restore(config['_checkpoint_restore'])

            for policy_source, policy_target in config['_restore_only']:
                source_keys = trainer_1.get_weights().keys()
                assert policy_source in source_keys, f"Wrong source key: {policy_source} {source_keys}"
                w = trainer_1.get_policy(policy_source).get_weights()
                target_keys = trainer.get_weights().keys()
                assert policy_target in target_keys, f"Wrong target key: {policy_target} {target_keys}"
                trainer.get_policy(policy_target).set_weights(w)
                logger.info(f"Set weights for policy {policy_target} from {config['_checkpoint_restore']}/{policy_source}")
        else:
            trainer_1 = get_trainer(config)
            trainer_1.restore(config['_checkpoint_restore'])
            trainer.set_weights(deepcopy(trainer_1.get_weights()))

    # restoring weights for specific policies
    if '_checkpoint_restore_policy' in config and iteration == 0:
        for policy, path in config['_checkpoint_restore_policy'].items():
            weights = pickle.load(open(path, 'rb'))
            trainer.get_policy(policy).set_weights(weights)
            logger.info(f"Set weights for policy {policy} from {path}")

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


def train_one_with_sacred(config, checkpoint_dir=None, **kwargs):
    """Train one trial with reporting to sacred."""
    del checkpoint_dir  # Unused, will look at the trainer dir and try to restore from .checkpoint_rllib
    checkpoint = None
    do_track = True
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
    ex.add_source_file('bursts.py')
    ex.add_source_file('helpers.py')
    ex.add_config(config=config, checkpoint=checkpoint, do_track=do_track, **config)

    @ex.main
    def train_one(config, checkpoint=None, do_track=True):
        """One trial with subprocesses for each iteration."""
        iteration = 0
        # trying to load the checkpoint...
        with tune.checkpoint_dir(step=0) as ckpt:
            ckpt = os.path.dirname(ckpt)
            try:
                def get_last_nonnull(df, attr):
                    """Get last value from a dataframe that is not null."""
                    if not hasattr(df, attr):
                        raise ValueError(f"Dataframe doesn't have an attribute {attr}")
                    arr = [x for x in arr if x]
                    if not arr:
                        raise ValueError(f"No non-null items in {arr}")
                    return arr[-1]

                df = get_df_from_logdir(ckpt)
                checkpoint_trainer = get_last_nonnull(df, attr='checkpoint_rllib')
                last_iteration = get_last_nonnull(df, attr='trainer_iteration')
                logger.info(f"Found previous run iteration={last_iteration} checkpoint={checkpoint_trainer}")
                checkpoint = checkpoint_trainer
                iteration = last_iteration
            except ValueError as err:
                logger.warning(f"Checkpoint loading for trial {ckpt} failed: {err}. Are there checkpoints?")

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

        # running iterations	
        while True:
            # doing edits in the config, bursts, for example
            config_updated = config
            if config['_update_config']:
                config_updated = config['_update_config'](config, iteration)
            config_updated['_iteration'] = iteration
            results = train_iteration(checkpoint, config_updated)
            checkpoint = results['checkpoint_rllib']
            iteration = results['trainer_iteration']
            logger.info("Iteration %d done" % iteration)

            # reporting
            dict_to_sacred(ex, results, iteration)

            if do_track:
                tune.report(**results)
            else:
                print(pretty_print(results))

            # stopping at the end
            if iteration >= config['_train_steps']:
                return

    ex.run()
    return None


def run_tune(config_name=None, config_override=None, tmp_dir=None, verbose=False, resume=False,
             show_only=False):
    """Call tune."""
    config = get_config_by_name(config_name)
    cluster_info = ray_init(tmp_dir=tmp_dir)
    tf.keras.backend.set_floatx('float32')

    # run metadata
    config['_main_filename'] = os.path.realpath(__file__)
    config['_redis_address'] = cluster_info['redis_address']
    config['_base_dir'] = os.path.dirname(os.path.realpath(__file__))
    config['_tmp_dir'] = tmp_dir
    config['_verbose'] = verbose

    # changing config entries from command line
    if config_override:
        config_override = json.loads(config_override)
        for k, v in config_override.items():
            config[k] = v

    if verbose:
        print("Template config")
        print(config)
        
    config['_call']['resume'] = resume #'PROMPT' if resume else False
    config['_call']['verbose'] = True
    config['_call']['queue_trials'] = True

    if show_only:
        print(pretty_print(config))
        return

    # running tune
    tune.run(
        train_one_with_sacred,
        config=config,
        **config['_call'],
    )


# main script: command-line interface
if __name__ == '__main__':
    args = parser.parse_args()

    # this option runs 1 training iteration
    if args.from_pickled_config:
        train_iteration_process(pickle_path=args.from_pickled_config)
        config = None

    # this option runs tune trials
    elif args.tune:
        config = args.tune
    else:
        config = select_config(title="Select main configuration to run")


    if config is not None:
        run_tune(config_name=config, config_override=args.config_override,
                 tmp_dir=args.tmp_dir, verbose=args.verbose, resume=args.resume,
                 show_only=args.show_config)

    sys.exit(0)
