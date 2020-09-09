import gym

class FlattenMultiagentEnvionment(gym.Env):
    """Flatten observations/actions of a multi-agent env."""
    def __init__(self, env_config):
        self._env = env_config['env']
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.policies = env_config['policies']
        self.players = []
        self.previous_obs = {}
        
    @staticmethod
    def dict_single_value(d):
        """Get single value in a dict."""
        assert len(d) == 1, f"Must have only 1 key, got {d.keys()}"
        return list(d.values())[0]
    
    def other_player_value(self, d, check_len=True, default=None):
        """Get value for the remaining player."""
        if check_len and default is None:
            assert len(d) - len(self.policies) == 1, f"Must have one value remaining. {d}"
        for k, v in d.items():
            if k not in self.policies:
                return v
        return default
        
    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)
        
    def reset(self):
        obs = self._env.reset()
        self.players = list(obs.keys())
        self.previous_obs = obs
        return self.other_player_value(obs)
    
    def step(self, action):
        actions = {}
        for p in self.players:
            if p in self.policies:
                policy = self.policies[p]
                actions[p] = policy(self.previous_obs[p])
            else:
                actions[p] = action
                
        obss, rews, dones, infos = self._env.step(actions)
        if obss:
            self.previous_obs = obss
        else:
            obss = self.previous_obs
                
        return self.other_player_value(obss), self.other_player_value(rews, default=np.array(0, dtype=np.float32)),\
               self.other_player_value(dones, check_len=False), self.other_player_value(infos, check_len=False)