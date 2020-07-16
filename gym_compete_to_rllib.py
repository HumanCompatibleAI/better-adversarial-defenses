from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gym
import numpy as np


class GymCompeteToRLLibAdapter(MultiAgentEnv):
    """Takes gym_compete env and makes a multi-agent RLLib env."""

    def __init__(self, env_cls, player_names=None):
        env = env_cls()
        super(GymCompeteToRLLibAdapter, self).__init__()
        #print(env.action_space)
        #print(env.observation_space)
        assert isinstance(env.action_space, gym.spaces.Tuple) and len(env.action_space) == 2
        assert isinstance(env.observation_space, gym.spaces.Tuple) and len(env.observation_space) == 2
        if player_names is None:
            player_names = ["player_%d" % i for i in range(1, 1 + len(env.action_space))]
        self.player_names = player_names
        self._env = env
        
        self.reset_dones()
        
    def reset_dones(self):
        self.dones = {name: False for name in self.player_names}
    
    def pack_array(self, array):
        """Given an array for players, return a dict player->value."""
        assert len(self.player_names) == len(array)
        return {p: o for p, o in zip(self.player_names, array)}
    
    def unpack_dict(self, dct, default_value=None):
        """Given a dict player->value return an array."""
        return [dct.get(p, default_value) for p in self.player_names]
    
    def reset(self):
        observations = self._env.reset()
        return self.pack_array(observations)

    def step(self, action_dict):
        #print(action_dict)
        default_action = np.zeros(self.observation_space.shape)
        a1a2 = self.unpack_dict(action_dict, default_action)
        o1o2, r1r2, done, i1i2 = self._env.step(a1a2)
        obs = self.pack_array(o1o2)
        rew = self.pack_array(r1r2)
        infos = self.pack_array([i1i2[0], i1i2[1]])
        dones = self.pack_array([i1i2[0]['agent_done'],
                                 i1i2[1]['agent_done']])
        #dones['__all__'] = done
        dones['__all__'] = done or all([dones[p] for p in self.player_names])
        
        # removing observations for already finished agents
        for p in self.player_names:
            if self.dones[p]:
                del obs[p]
                del rew[p]
                del infos[p]
        
        for p in self.player_names:
            self.dones[p] = dones[p]
        #print('dones', dones)
        return obs, rew, dones, infos
    
    @property
    def observation_space(self):
        # for one agent
        return self._env.observation_space[0]
    
    @property
    def action_space(self):
        # for one agent
        return self._env.action_space[0]