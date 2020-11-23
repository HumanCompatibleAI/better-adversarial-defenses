import os
import pickle
import uuid
from copy import deepcopy
from typing import List

import ray
from ray.rllib.agents import with_common_config
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as config_ppo
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches, StandardizeFields, SelectExperiences
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, MultiAgentBatch

from ap_rllib.helpers import filter_pickleable, dict_get_any_value, save_gym_space, unlink_ignore_error
from frankenstein.remote_communicator import RemoteHTTPPickleCommunicator
from gym_compete_rllib.load_gym_compete_policy import nets_to_weights, load_weights_from_vars


def rllib_samples_to_dict(samples):
    """Convert rllib MultiAgentBatch to a dict."""
    samples = samples.policy_batches
    samples = {x: dict(y) for x, y in samples.items()}
    return samples


def train_external(policies, samples, config):
    """Train using a TCP stable_baselines server, return info."""
    infos = {}
    answer_paths = {}
    data_paths = {}

    # doing nothing for make_video.py
    if config['lr'] == 0:
        return {}

    samples_dict = rllib_samples_to_dict(samples)

    # only training policies with data
    to_train = set(policies)
    to_train = to_train.intersection(samples_dict.keys())

    config_orig = deepcopy(config)
    config = filter_pickleable(config_orig)

    # config to send
    p = dict_get_any_value(config_orig['multiagent']['policies'])
    print(config_orig['multiagent']['policies'])
    obs_space, act_space = p[1], p[2]
    config['_observation_space'] = save_gym_space(obs_space)
    config['_action_space'] = save_gym_space(act_space)

    communicator = RemoteHTTPPickleCommunicator(config['http_remote_port'])

    # requesting to train all policies
    for policy in to_train:
        # only training the requested policies
        if policy not in config['multiagent']['policies_to_train']:
            continue

        # identifier for this run
        run_uid = config['run_uid']

        # identifier for the run+policy
        run_policy_uid = f"{run_uid}_policy_{policy}"

        # unique step information
        iteration = str(uuid.uuid1())

        # identifier for run+policy_current step
        run_policy_step_uid = f"{run_uid}_policy_{policy}_step{iteration}"

        # data to pickle
        data_policy = {'rollouts': samples_dict[policy],
                       'weights': nets_to_weights(policies[policy].model._nets),
                       'config': config}

        # paths for data/answer
        tmp_dir = os.getcwd()  # config['tmp_dir']
        data_path = os.path.join(tmp_dir, run_policy_step_uid + '.pkl')
        answer_path = os.path.join(tmp_dir, run_policy_step_uid + '.answer.pkl')
        data_paths[policy] = data_path
        answer_paths[policy] = answer_path

        # saving pickle data
        pickle.dump(data_policy, open(data_path, 'wb'))

        # connecting to the RPC server
        communicator.submit_job(client_id=run_policy_uid, data_path=data_path,
                                answer_path=answer_path, data=data_policy)

    # obtaining policies
    for policy in to_train:
        answer_path = answer_paths[policy]
        data_path = data_paths[policy]

        weights_info = communicator.get_result(answer_path)

        # checking correctness
        if not (weights_info[0] is True):
            raise Exception(weights_info[1])

        weights = weights_info[1]['weights']
        info = weights_info[1]['info']

        def load_weights(model, weights):
            """Load weights into a model."""
            load_weights_from_vars(weights, model._nets['value'], model._nets['policy'])

        # loading weights into the model
        load_weights(policies[policy].model, weights)

        # removing pickle files to save space
        unlink_ignore_error(data_path)
        unlink_ignore_error(answer_path)

        # obtaining info
        infos[policy] = dict(info)

    return infos
