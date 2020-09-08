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
from stable_baselines.logger import KVWriter, SeqWriter, DEBUG


class dummy_env(object):
    """Dummy environment to give something as output."""
    def __init__(self, env_orig):
        self.metadata = {}
        self.observation_space = env_orig.observation_space[0]
        self.action_space = env_orig.action_space[0]
        
        self.episode_count = 0
        self.episode_length = 100
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        self.episode_count += 1
        return np.ones(380) * self.episode_count
    
    def step(self, action):
        obs = np.ones(380) * self.episode_count
        #rew = np.random.rand()
        rew = 1. * (self.current_step == self.episode_length - 1)
        info = {}
        self.current_step += 1
        done = self.current_step >= self.episode_length
        return obs, rew, done, info
    
    
    
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


class mock_vanilla_runner(AbstractEnvRunner):
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
        

    def _run(self):
        # mb stands for minibatch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)

            self.model.num_timesteps += self.n_envs

            if self.callback is not None:
                # Abort training early
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 9

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)
                        
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        
        
        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))
        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward
            
    
class ConstDataRunner(AbstractEnvRunner):
    def __init__(self, rollout, *, env=None, model=None, n_steps=None, gamma=0.9, lam=1):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.lam = lam
        self.gamma = gamma
        self.rollout = rollout
        
        # sanity checks: all keys have same length
        for k in rollout.keys():
            assert len(rollout[k]) == len(rollout['t'])

        # sanity checks on rollout data: episode consistency
        prev_data = None
        for i in range(len(rollout['t'])):
            data = {k: rollout[k][i] for k in rollout.keys()}
            if prev_data:
                if data['eps_id'] != prev_data['eps_id']:
                    assert prev_data['dones']
                else:
                    assert data['t'] > prev_data['t']
            prev_data = deepcopy(data)
            
        # sanity check: number of episodes equals number of DONEs
        assert np.abs(np.sum(rollout['dones']) - len(set(rollout['eps_id']))) <= 1
        self.run_calls = 0
        
    def _run(self):
        """
        Run a learning step of the model.
        """
        
        # only using data once
        if self.run_calls:
            print('Calling run() twice')
        # assert self.run_calls == 0
        self.run_calls += 1
        
        states = None
        ep_infos = []
        rollout = self.rollout
        obs = rollout['obs']
        masks = rollout['dones']
        actions = rollout['actions']
        values = rollout['vf_preds']
        true_reward = rollout['rewards']
        neglogpacs = -rollout['action_logp']
        advantages = np.zeros_like(true_reward)
        
        last_gae_lam = 0
        
        n_steps = len(obs)
        
        self.model.num_timesteps += n_steps
        
        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                nextnonterminal = 1.0 - masks[-1]
                nextvalues = values[-1]
            else:
                nextnonterminal = 1.0 - masks[step + 1]
                nextvalues = values[step + 1]

            delta = true_reward[step] + self.gamma * nextvalues * nextnonterminal - values[step]
            advantages[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
            #print(step, values[step], true_reward[step], delta, advantages[step], advantages[step] + values[step])
        
        returns = advantages + values
        return obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward
    
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
        self.env_orig = gym.make(config['env_config']['env_name'])
        self.env = dummy_env(self.env_orig)
        self.ppo = PPO(policy=policy.MlpPolicyValue, env=self.env, policy_kwargs={'normalize': True}, n_steps=100,
                       **dict(gamma=config['gamma'], lam=config['lambda'], ent_coef=config['entropy_coeff'],
                              learning_rate=config['lr'], vf_coef=config['vf_loss_coeff'],
                              max_grad_norm=config['grad_clip'], nminibatches=config['train_batch_size']//config['sgd_minibatch_size'],
                              noptepochs=config['num_sgd_iter'], cliprange=config['clip_param']))
        
        self.logged_data = []
        self.nminibatches = config['train_batch_size']//config['sgd_minibatch_size']
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