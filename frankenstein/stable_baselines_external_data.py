from copy import deepcopy

import numpy as np
from gym_compete import policy
from stable_baselines import PPO2 as PPO
from stable_baselines.common import callbacks
from stable_baselines.logger import KVWriter, SeqWriter, DEBUG
from stable_baselines.ppo2.ppo2 import Runner as PPORunner
from ap_rllib.helpers import load_gym_space
from frankenstein.dummy_env import dummy_env
from frankenstein.dummy_ppo_runner import mock_vanilla_runner
from frankenstein.ram_logger import RAMFormat, LoggerOnlyLogCallback


class SBPPORemoteData(object):
    """Run PPO from stable baselines from given data."""

    def __init__(self, config):
        """Create the trainer, and logger."""
        # dummy environment with nothing but shapes
        self.env = dummy_env(config=config)
        
        # the stable baselines trainer
        self.ppo = PPO(policy=policy.MlpPolicyValue, env=self.env, policy_kwargs={'normalize': True}, n_steps=100,
                       **dict(gamma=config['gamma'], lam=config['lambda'], ent_coef=config['entropy_coeff'],
                              learning_rate=config['lr'], vf_coef=config['vf_loss_coeff'],
                              max_grad_norm=config['grad_clip'],
                              nminibatches=config['train_batch_size'] // config['sgd_minibatch_size'],
                              noptepochs=config['num_sgd_iter'], cliprange=config['clip_param']))
        
        # number of minibatches for stable baselines
        self.nminibatches = config['train_batch_size'] // config['sgd_minibatch_size']
        
        # buffer for logged metrics
        self.logged_data = []
        
        # logger to collect metrics
        self.logger_format = RAMFormat(self.logged_data)
        self.log_callback = LoggerOnlyLogCallback(self.logger_format)
        
        # the policy name in stable baselines
        # used to get/set the weights
        self.prefix = list(self.ppo.get_parameters().keys())[0].split('/')[0] + '/'

    def set_weights(self, weights):
        """Set weights for the policy."""
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
        """Get weights of the policy."""
        return {x[len(self.prefix):]: y for x, y in self.ppo.get_parameters().items()}

    def learn(self, rllib_rollout_):
        """Run training with stable baselines."""
        # total length of the rollout
        T = len(rllib_rollout_['t'])
        
        # 1) making sure that T is divisable by nminibatches
        # 2) removing the last minibatch (trainer actually accesses one more)
        T = T - T % self.nminibatches - self.nminibatches
        
        assert T > 0, "Size of the rollouts is too small, need to be at least 1 minibatch."
        
        # making sure the rollouts are of proper length
        rllib_rollout = {x: y[:T + 1] for x, y in rllib_rollout_.items()}
        
        # giving parameters of the data:
        # number of steos samples
        self.ppo.n_steps = T
        
        # print additional info
        self.ppo.verbose = 1
        
        # number of items inside one batch (!=minibatch)
        self.ppo.n_batch = T
        
        # creating an object which gives the data from the buffer
        # and has the same interface as the actual runner
        r1 = mock_vanilla_runner(rllib_rollout, env=self.ppo.env, n_steps=self.ppo.n_steps, model=self.ppo,
                                 gamma=self.ppo.gamma, lam=self.ppo.lam)
        
        # setting the runner
        self.ppo._runner = r1
        
        # running the leearning
        self.ppo.learn(total_timesteps=T, callback=self.log_callback)
        
        # returning the logged data
        metrics = self.logged_data[-1]

        # clearing the buffer to free memory
        self.logged_data.clear()
        
        return metrics