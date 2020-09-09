#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
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
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG

from ray.rllib.models import ModelCatalog
import uuid


import ray
from ray import tune
from ray.tune import track

import math
import gym

import gym_compete_rllib.gym_compete_to_rllib
from gym_compete_rllib.gym_compete_to_rllib import created_envs, create_env
from gym_compete_rllib.load_gym_compete_policy import nets_to_weight_array, nets_to_weights, load_weights_from_vars

import pickle
from copy import deepcopy
from jsonrpcclient.clients.http_client import HTTPClient


import os


from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents import with_common_config
from collections import defaultdict
import logging
import numpy as np
import math
from typing import List

import ray
from ray.rllib.evaluation.metrics import get_learner_stats, LEARNER_STATS_KEY
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.common import SampleBatchType,     STEPS_SAMPLED_COUNTER, STEPS_TRAINED_COUNTER, LEARNER_INFO,     APPLY_GRADS_TIMER, COMPUTE_GRADS_TIMER, WORKER_UPDATE_TIMER,     LEARN_ON_BATCH_TIMER, LOAD_BATCH_TIMER, LAST_TARGET_UPDATE_TS,     NUM_TARGET_UPDATES, _get_global_vars, _check_sample_batch_type,     _get_shared_metrics
from ray.rllib.execution.multi_gpu_impl import LocalSyncParallelOptimizer
from ray.rllib.policy.policy import PolicyID
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID,     MultiAgentBatch
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.sgd import do_minibatch_sgd, averaged

from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as config_ppo

from ray.rllib.agents import with_common_config
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches,     StandardizeFields, SelectExperiences
from ray.rllib.execution.train_ops import TrainOneStep, TrainTFMultiGPU
from ray.rllib.execution.metric_ops import StandardMetricsReporting

def rllib_samples_to_dict(samples):
    """Convert rllib MultiAgentBatch to a dict."""
    samples = samples.policy_batches
    samples = {x: dict(y) for x, y in samples.items()}
    return samples


def train_external(policies, samples, config):
    """Train using a TCP stable_baselines server."""
    infos = {}
    
    for policy in policies:
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
        data_policy = {}
        
        # config to send
        config_orig = deepcopy(config)
        config = filter_dict_pickleable(config)
        p = dict_get_any_value(config['multiagent']['policies'])
        obs_space, act_space = p[1], p[2]
        config['_observation_space'] = obs_space
        config['_action_space'] = act_space
        
        # data: rollouts and weights
        data_policy['rollouts'] = rllib_samples_to_dict(samples)[policy]
        data_policy['weights'] = nets_to_weights(policies[policy].model._nets)
        data_policy['config'] = config

        # paths for data/answer
        data_path = run_policy_step_uid + '.pkl'
        answer_path = run_policy_step_uid + '_answer.pkl'
        
        # saving pickle data
        pickle.dump(data_policy, open(data_path, 'wb'))

        # connecting to the RPC server
        client = HTTPClient(config['http_remote_port'])
        result = client.process(run_policy_uid, uid=0, config=config, data_path=data_path, answer_path=answer_path).data.result
        
        # checking for result correctness
        if result != True:
            raise ValueError("Wrong result", str(result))

        # loading weights and information
        weights_info = pickle.load(open(answer_path, 'rb'))
        weights = weights_info['weights']
        info = weights_info['info']

        # loading weights into the model
        def load_weights(model, weights):
            """Load weights into a model."""
            load_weights_from_vars(weights, model._nets['value'], model._nets['policy'])
        load_weights(policies[policy].model, weights)

        # removing pickle files to save space
        unlink_ignore_error(data_path)
        unlink_ignore_error(answer_path)
        
        infos[policy] = (dict(info))
        
    return infos

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


# In[32]:


DEFAULT_CONFIG = deepcopy(config_ppo)
DEFAULT_CONFIG.update({'http_remote_port': "http://127.0.0.1:50001", 'run_uid': 'aba'})
DEFAULT_CONFIG = with_common_config(DEFAULT_CONFIG)

ExternalTrainer = build_trainer(
    name="External",
    default_config=DEFAULT_CONFIG,
    default_policy=PPOTFPolicy,
    execution_plan=execution_plan)

