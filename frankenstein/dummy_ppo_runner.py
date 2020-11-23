from copy import deepcopy
import numpy as np
from gym_compete import policy
from stable_baselines import PPO2 as PPO
from stable_baselines.common import callbacks
from stable_baselines.logger import KVWriter, SeqWriter, DEBUG
from stable_baselines.ppo2.ppo2 import Runner as PPORunner
from ap_rllib.helpers import load_gym_space


class mock_vanilla_runner(PPORunner):
    """Runner class for Stable Baselines PPO sampling from a pre-defined buffer instead of the environment."""
    
    def __init__(self, rollout, *, env=None, model=None, n_steps=None, gamma=0.9, lam=1):
        self.rollouts = rollout
        self.states = None
        self.dones = self.rollouts['dones'][0:1]
        self.true_env = env
        self.true_model = model
        self.gamma = gamma
        self.lam = lam

        class model_cls(object):
            def __init__(self, rollouts, true_model):
                self.rollouts = rollouts
                self.true_model = true_model
                self.idx = 0
                self.num_timesteps = 0

            def step(self, obs, states, dones):
                states = None
                actions = self.rollouts['actions'][self.idx]
                values = self.rollouts['vf_preds'][self.idx]
                neglogpacs = -self.rollouts['action_logp'][self.idx]
                self.idx += 1
                return np.array([actions]), np.array([values]), np.array([states]), np.array([neglogpacs])

            def value(self, obs, states, dones):
                return np.array([self.rollouts['vf_preds'][-1]])

        class env_cls(object):
            def __init__(self, rollouts, true_env):
                self.true_env = true_env
                self.rollouts = rollouts
                self.action_space = true_env.action_space
                self.observation_space = true_env.observation_space
                self.idx = 0

            def reset(self):
                self.idx += 1
                return np.array([self.rollouts['obs'][0]])

            def step(self, actions):
                obs = self.rollouts['obs'][self.idx]
                rew = self.rollouts['rewards'][self.idx]
                done = self.rollouts['dones'][self.idx]
                info = self.rollouts['infos'][self.idx]
                self.idx += 1
                return np.array([obs]), np.array([rew]), np.array([done]), np.array([info])

        self.model = model_cls(self.rollouts, self.true_model)
        self.env = env_cls(self.rollouts, self.true_env)

        self.obs = self.env.reset()
        self.n_envs = 1
        self.callback = None
        self.n_steps = len(self.rollouts['obs']) - 1