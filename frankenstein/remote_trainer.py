#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import uuid
from copy import deepcopy
from time import sleep
from typing import List

import ray
from jsonrpcclient.clients.http_client import HTTPClient
from ray.rllib.agents import with_common_config
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as config_ppo
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.common import SampleBatchType, STEPS_TRAINED_COUNTER, LEARNER_INFO, \
    WORKER_UPDATE_TIMER, LEARN_ON_BATCH_TIMER, LOAD_BATCH_TIMER, \
    _get_global_vars, _check_sample_batch_type, _get_shared_metrics
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches, StandardizeFields, SelectExperiences
from ray.rllib.policy.policy import PolicyID
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, MultiAgentBatch

from gym_compete_rllib.load_gym_compete_policy import nets_to_weights, load_weights_from_vars
from ap_rllib.helpers import filter_pickleable, dict_get_any_value, save_gym_space, unlink_ignore_error


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
        data_path = run_policy_step_uid + '.pkl'
        answer_path = run_policy_step_uid + '.answer.pkl'
        data_paths[policy] = data_path
        answer_paths[policy] = answer_path

        # saving pickle data
        pickle.dump(data_policy, open(data_path, 'wb'))

        # connecting to the RPC server
        client = HTTPClient(config['http_remote_port'])
        result = client.process(run_policy_uid, uid=0, data_path=data_path, answer_path=answer_path).data.result

        assert result is True, str(result)

    # obtaining policies
    for policy in to_train:
        answer_path = answer_paths[policy]
        data_path = data_paths[policy]

        # loading weights and information
        # busy wait with a delay
        while True:
            try:
                weights_info = pickle.load(open(answer_path, 'rb'))
                break
            except Exception as e:
                print(e, "Waiting")
                sleep(0.5)

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


# copied from TrainTFMultiGPU and modified
class ExternalTrainOp:
    """Train using the function above externally."""

    def __init__(self,
                 workers: WorkerSet,
                 config: dict,
                 policies: List[PolicyID] = frozenset([])):
        self.workers = workers
        self.policies = policies or workers.local_worker().policies_to_train
        self.config = config

    def __call__(self,
                 samples: SampleBatchType) -> (SampleBatchType, List[dict]):
        _check_sample_batch_type(samples)

        # Handle everything as if multiagent
        if isinstance(samples, SampleBatch):
            samples = MultiAgentBatch({
                DEFAULT_POLICY_ID: samples
            }, samples.count)

        # data: samples

        metrics = _get_shared_metrics()
        load_timer = metrics.timers[LOAD_BATCH_TIMER]
        learn_timer = metrics.timers[LEARN_ON_BATCH_TIMER]

        # calling train_external to train with stable baselines
        p = {k: self.workers.local_worker().get_policy(k) for k in self.policies}
        info = train_external(policies=p, samples=samples, config=self.config)

        load_timer.push_units_processed(samples.count)
        learn_timer.push_units_processed(samples.count)

        fetches = info

        metrics.counters[STEPS_TRAINED_COUNTER] += samples.count
        metrics.info[LEARNER_INFO] = fetches
        if self.workers.remote_workers():
            with metrics.timers[WORKER_UPDATE_TIMER]:
                weights = ray.put(self.workers.local_worker().get_weights(
                    self.policies))
                for e in self.workers.remote_workers():
                    e.set_weights.remote(weights, _get_global_vars())
        # Also update global vars of the local worker.
        self.workers.local_worker().set_global_vars(_get_global_vars())
        return samples, fetches


def execution_plan(workers, config):
    """Execution plan which calls ExternalTrainOp."""
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # Collect large batches of relevant experiences & standardize.
    rollouts = rollouts.for_each(
        SelectExperiences(workers.trainable_policies()))
    rollouts = rollouts.combine(
        ConcatBatches(min_batch_size=config["train_batch_size"]))
    rollouts = rollouts.for_each(StandardizeFields(["advantages"]))

    train_op = rollouts.for_each(
        ExternalTrainOp(workers=workers,
                        config=config))

    return StandardMetricsReporting(train_op, workers, config)


# creating ExternalTrainer
DEFAULT_CONFIG = deepcopy(config_ppo)
DEFAULT_CONFIG.update({'http_remote_port': "http://127.0.0.1:50001", 'run_uid': 'aba'})
DEFAULT_CONFIG = with_common_config(DEFAULT_CONFIG)

ExternalTrainer = build_trainer(
    name="External",
    default_config=DEFAULT_CONFIG,
    default_policy=PPOTFPolicy,
    execution_plan=execution_plan)
