from gym.spaces import Box
from ray.rllib.examples.env.rock_paper_scissors import RockPaperScissors
import numpy as np


class RPSNoise(RockPaperScissors):
    """RockPaperScissors with noise as observation."""
    
    def __init__(self, config):
        self.noise_dim = config.get("noise_dim", 4)        
        
        super(RPSNoise, self).__init__(config)
        self.observation_space = Box(np.full(self.noise_dim, -np.inf),
                                     np.full(self.noise_dim, np.inf))
    
    def _sample_noise(self):
        return np.random.randn(self.noise_dim)
    
    def _transform_obs(self, obs):
        noise = self._sample_noise()
        return {x: noise for x, y in obs.items()}
    
    def reset(self):
        obs = super(RPSNoise, self).reset()
        obs = self._transform_obs(obs)
        return obs
    
    def step(self, action_dict):
        obs, rew, done, info = super(RPSNoise, self).step(action_dict)
        obs = self._transform_obs(obs)
        return obs, rew, done, info