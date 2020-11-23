from copy import deepcopy

import numpy as np
from gym_compete import policy
from stable_baselines import PPO2 as PPO
from stable_baselines.common import callbacks
from stable_baselines.logger import KVWriter, SeqWriter, DEBUG
from stable_baselines.ppo2.ppo2 import Runner as PPORunner
from ap_rllib.helpers import load_gym_space


class dummy_env(object):
    """Dummy environment, loads gym spaces from the config."""

    def __init__(self, config):
        self.metadata = {}
        self.observation_space = load_gym_space(config['_observation_space'])
        self.action_space = load_gym_space(config['_action_space'])