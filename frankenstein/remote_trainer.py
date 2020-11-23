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
from ray.rllib.execution.common import SampleBatchType, STEPS_TRAINED_COUNTER, LEARNER_INFO, \
    WORKER_UPDATE_TIMER, LEARN_ON_BATCH_TIMER, LOAD_BATCH_TIMER, \
    _get_global_vars, _check_sample_batch_type, _get_shared_metrics
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches, StandardizeFields, SelectExperiences
from ray.rllib.policy.policy import PolicyID
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, MultiAgentBatch

from ap_rllib.helpers import filter_pickleable, dict_get_any_value, save_gym_space, unlink_ignore_error
from frankenstein.remote_communicator import RemoteHTTPPickleCommunicator
from gym_compete_rllib.load_gym_compete_policy import nets_to_weights, load_weights_from_vars

# the function that does the training
from frankenstein.remote_trainer_with_communicator import train_external


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

# default values, can be changed using rllib configuration
DEFAULT_CONFIG.update({'http_remote_port': "http://127.0.0.1:50001", 'run_uid': 'aba', 'tmp_dir': '/tmp/'})
DEFAULT_CONFIG = with_common_config(DEFAULT_CONFIG)

ExternalTrainer = build_trainer(
    name="External",
    default_config=DEFAULT_CONFIG,
    default_policy=PPOTFPolicy,
    execution_plan=execution_plan)
