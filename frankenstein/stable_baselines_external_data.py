import pickle
import numpy as np
from matplotlib import pyplot as plt
import gym_compete
import gym
from gym_compete import policy
import tensorflow as tf
import uuid
from copy import deepcopy
import stable_baselines
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines import PPO2 as PPO
from stable_baselines import logger
from stable_baselines.common import callbacks
from helpers import load_gym_space
from stable_baselines.logger import KVWriter, SeqWriter, DEBUG
from stable_baselines.ppo2.ppo2 import Runner as PPORunner


class dummy_env(object):
    """Dummy environment to give something as output."""
    def __init__(self, config):
        self.metadata = {}
        self.observation_space = load_gym_space(config['_observation_space'])
        self.action_space = load_gym_space(config['_action_space'])

class mock_vanilla_runner(PPORunner):
    #def __init__(self, rollouts, weights, env, true_model, gamma, lam):
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
                #value_true = self.true_model.value(np.array([obs]))[0]
                #actions_unclip = self.rollouts['actions'][self.idx-1]
                #actions_unclip = self.rollouts['actions_unclipped'][self.idx]
                #neglogpac_true = -np.log(sbppo.ppo.action_probability(obs, actions=actions_unclip)[0])
                #if np.linalg.norm(values - value_true) > 1e-5:
                #    print("Wrong value", values, value_true)

                neglogpacs = -self.rollouts['action_logp'][self.idx]
                #neglogpacs = self.rollouts['true_neglogp'][self.idx]

                #if np.linalg.norm(neglogpacs - neglogpac_true) > 1e-3 and dones:
                #    print("Wrong NegLogP", neglogpacs, neglogpac_true)

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
                #print(actions, self.rollouts['actions'][self.idx - 1])
                #assert np.allclose(actions, self.rollouts['actions'][self.idx - 1])
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

    
class RAMFormat(KVWriter, SeqWriter):
    def __init__(self, arr):
        self.arr = arr
    
    def writekvs(self, kvs):
        self.arr.append(deepcopy(kvs))
        
class LoggerOnlyLogCallback(callbacks.BaseCallback):

    def __init__(self, log_ram_format, *args, **kwargs):
        super(LoggerOnlyLogCallback, self).__init__(*args, **kwargs)
        self.log_ram_format = log_ram_format
        
    def _on_training_start(self):
        self.logger.level = DEBUG
        if self.log_ram_format not in self.logger.output_formats:
            self.logger.output_formats.append(self.log_ram_format)    
    
class SBPPORemoteData(object):
    """Run PPO from stable baselines from given data."""
    def __init__(self, config):
        self.env = dummy_env(config=config)
        self.ppo = PPO(policy=policy.MlpPolicyValue, env=self.env, policy_kwargs={'normalize': True}, n_steps=100,
                       **dict(gamma=config['gamma'], lam=config['lambda'], ent_coef=config['entropy_coeff'],
                              learning_rate=config['lr'], vf_coef=config['vf_loss_coeff'],
                              max_grad_norm=config['grad_clip'], nminibatches=config['train_batch_size']//config['sgd_minibatch_size'],
                              noptepochs=config['num_sgd_iter'], cliprange=config['clip_param']))
        
        self.logged_data = []
        self.nminibatches = config['train_batch_size'] // config['sgd_minibatch_size']
        self.logger_format = RAMFormat(self.logged_data)
        self.log_callback = LoggerOnlyLogCallback(self.logger_format)
        self.prefix = list(self.ppo.get_parameters().keys())[0].split('/')[0] + '/'
        
    def set_weights(self, weights):
        # loading weights
        prefix = self.prefix

        w1 = self.ppo.get_parameters()
        assert set([x[len(prefix):] for x in w1.keys()]) == set(weights.keys())

        w_out = {}
        for k in weights.keys():
            assert weights[k].shape == w1[prefix + k].shape
            w_out[prefix + k] = weights[k]
        self.ppo.load_parameters(w_out)
        
    def get_weights(self):
        return {x[len(self.prefix):]: y for x, y in self.ppo.get_parameters().items()}
        
    def learn(self, rllib_rollout_):
        T = len(rllib_rollout_['t'])
        T = T - T % self.nminibatches - self.nminibatches
        rllib_rollout = {x: y[:T+1] for x, y in rllib_rollout_.items()}
        self.ppo.n_steps = T
        self.ppo.verbose = 1
        self.ppo.n_batch = T
        #r1 = ConstDataRunner(rllib_rollout, env=self.ppo.env, n_steps=self.ppo.n_steps, model=self.ppo,
        #                     gamma=self.ppo.gamma, lam=self.ppo.lam)
        r1 = mock_vanilla_runner(rllib_rollout, env=self.ppo.env, n_steps=self.ppo.n_steps, model=self.ppo,
                                 gamma=self.ppo.gamma, lam=self.ppo.lam)
        self.ppo._runner = r1
        self.ppo.learn(total_timesteps=T, callback=self.log_callback)
        return self.logged_data[-1]
