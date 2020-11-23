from copy import deepcopy
import numpy as np
from gym_compete import policy
from stable_baselines import PPO2 as PPO
from stable_baselines.common import callbacks
from stable_baselines.logger import KVWriter, SeqWriter, DEBUG
from stable_baselines.ppo2.ppo2 import Runner as PPORunner
from ap_rllib.helpers import load_gym_space


class RAMFormat(KVWriter, SeqWriter):
    """Logger which stores messages inside the memory."""
    def __init__(self, arr):
        self.arr = arr

    def writekvs(self, kvs):
        self.arr.append(deepcopy(kvs))


class LoggerOnlyLogCallback(callbacks.BaseCallback):
    """Save training data into logger."""
    def __init__(self, log_ram_format, *args, **kwargs):
        super(LoggerOnlyLogCallback, self).__init__(*args, **kwargs)
        self.log_ram_format = log_ram_format

    def _on_training_start(self):
        self.logger.level = DEBUG
        if self.log_ram_format not in self.logger.output_formats:
            self.logger.output_formats.append(self.log_ram_format)
