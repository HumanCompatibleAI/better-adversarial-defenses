from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from ray.tune.registry import register_env
from copy import deepcopy
from gym.spaces import Box

class DummyMultiAgentEnv(MultiAgentEnv):
    """Does not do anything, used for testing."""

    def __init__(self, config=None):
        super(DummyMultiAgentEnv, self).__init__()
        config_default = dict(action_dim=17, obs_dim=380, n_players=2, n_steps=1000)
        self.config = deepcopy(config_default)
        for k, v in config.items():
            self.config[k] = v
        self.action_dim = self.config.get('action_dim')
        self.obs_dim = self.config.get('obs_dim')
        self.n_players = self.config.get('n_players')
        self.n_steps = self.config.get('n_steps')
        self.player_names = ["player_%d" % p for p in range(1,  1 + self.n_players)]
        self.n_step = 0
        
    def _get_random_observation(self):
        return np.random.randn(self.obs_dim)

    def reset(self):
        self.n_step = 0
        return {p: self._get_random_observation() for p in self.player_names}

    def step(self, action_dict):
        obs = {p: self._get_random_observation() for p in self.player_names}
        rew = {p: 0.0 for p in self.player_names}
        done = self.n_step >= self.n_steps
        dones = {p: done for p in self.player_names}
        dones['__all__'] = done
        infos = {p: {} for p in self.player_names}

        return obs, rew, dones, infos

    @property
    def observation_space(self):
        high = np.full((self.obs_dim,), fill_value=np.inf)
        return Box(low=-high, high=high)

    @property
    def action_space(self):
        high = np.full((self.action_dim,), fill_value=np.inf)
        return Box(low=-high, high=high)

def create_env(config):
    return DummyMultiAgentEnv(config)

env_name = "DummyMultiAgentEnv"
register_env(env_name, create_env)