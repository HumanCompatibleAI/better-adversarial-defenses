import numpy as np
import ray
import tensorflow as tf
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from gym.spaces import Box
from ray.tune.registry import register_env


def dim_to_gym_box(dim, val=np.inf):
    """Create gym.Box with specified dimension."""
    high = np.full((dim,), fill_value=val)
    return Box(low=-high, high=high)


class DummyMultiAgentEnv(MultiAgentEnv):
    """Return zero observations."""

    def __init__(self, config):
        del config  # Unused
        super(DummyMultiAgentEnv, self).__init__()
        self.config = dict(act_dim=17, obs_dim=380, n_players=2, n_steps=1000)
        self.players = ["player_%d" % p for p in range(self.config['n_players'])]
        self.current_step = 0

    def _obs(self):
        return np.zeros((self.config['obs_dim'],))

    def reset(self):
        self.current_step = 0
        return {p: self._obs() for p in self.players}

    def step(self, action_dict):
        done = self.current_step >= self.config['n_steps']
        self.current_step += 1

        obs = {p: self._obs() for p in self.players}
        rew = {p: 0.0 for p in self.players}
        dones = {p: done for p in self.players + ["__all__"]}
        infos = {p: {} for p in self.players}

        return obs, rew, dones, infos

    @property
    def observation_space(self):
        return dim_to_gym_box(self.config['obs_dim'])

    @property
    def action_space(self):
        return dim_to_gym_box(self.config['act_dim'])


def create_env(config):
    """Create the dummy environment."""
    return DummyMultiAgentEnv(config)


env_name = "DummyMultiAgentEnv"
register_env(env_name, create_env)


def get_trainer_config(env_config, train_policies, num_workers=5, framework="tfe"):
    """Build configuration for 1 run."""

    # obtaining parameters from the environment
    env = create_env(env_config)
    act_space = env.action_space
    obs_space = env.observation_space
    players = env.players
    del env

    def get_policy_config(p):
        """Get policy configuration for a player."""
        return (PPOTFPolicy, obs_space, act_space, {
            "model": {
                "use_lstm": False,
                "fcnet_hiddens": [64, 64],
            },
            "framework": framework,
        })

    # creating policies
    policies = {p: get_policy_config(p) for p in players}

    # trainer config
    config = {
        "env": env_name, "env_config": env_config, "num_workers": num_workers,
        "multiagent": {"policy_mapping_fn": lambda x: x, "policies": policies,
                       "policies_to_train": train_policies},
        "framework": framework,
        "train_batch_size": 32768,
        "num_sgd_iter": 1,
        "sgd_minibatch_size": 32768,

        # 450 megabytes for each worker (enough for 1 iteration)
        "memory_per_worker": 450 * 1024 * 1024,
        "object_store_memory_per_worker": 128 * 1024 * 1024,
    }
    return config


def tune_run():
    ray.init(ignore_reinit_error=True)
    config = get_trainer_config(train_policies=['player_1', 'player_2'], env_config={})
    return tune.run("PPO", config=config, verbose=True, name="dummy_run", num_samples=1)


if __name__ == '__main__':
    tune_run()
