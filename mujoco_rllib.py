import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
from ray.tune.registry import register_env, _global_registry, ENV_CREATOR
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

import ray
from ray import tune
from ray.tune import track

import math

ray.init(ignore_reinit_error=True, include_webui=True,
                            )#temp_dir='/scratch/sergei/tmp')


trainer = PPOTrainer(config={'train_batch_size': 4000,
    'env': 'Reacher-v2', 'num_workers': 0})

for _ in range(10):
    res = trainer.train()
    print(pretty_print(res))
