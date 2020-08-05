import numpy as np
import ray
import tensorflow as tf
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.logger import pretty_print
import numpy as np
from gym.spaces import Box
from ray.tune.registry import register_env
import gc
import multiprocessing


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
        #objects = gc.get_objects()
        #objects = map(lambda x : str(type(x)), objects)
        #objects = filter(lambda x: x.startswith("""<class 'ray."""), objects)
        #objects = list(objects)
        #objects_unique = np.unique(objects)
        #objects = {o: len([x for x in objects if x == o]) for o in objects_unique}
        #for k, v in objects.items():
        #    print(k, v)
        #gc.collect()

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


def get_trainer_config(env_config, train_policies, num_workers=1, framework="tfe"):
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

    bs = 1005
    # trainer config
    config = {
        "env": env_name, "env_config": env_config, "num_workers": num_workers,
        "multiagent": {"policy_mapping_fn": lambda x: x, "policies": policies,
                       "policies_to_train": train_policies},
        "framework": framework,
        "train_batch_size": bs,
        "num_sgd_iter": 1,
        "sgd_minibatch_size": bs,

        # 450 megabytes for each worker (enough for 1 iteration)
        "memory_per_worker": 450 * 1024 * 1024,
        "object_store_memory_per_worker": 128 * 1024 * 1024,
    }
    return config

def recreate_remote_workers(self):
    """Recreate workers in a worker set."""
    print("RECREATE WORKERS")
    #traceback.print_stack()
    num_workers = len(self._remote_workers)
    for w in self._remote_workers:
        w.stop.remote()
        del w
        #w.__ray_terminate__.remote()

    self._remote_workers = []
    self.add_workers(num_workers)

def run_ppo(config, restore_state=None):
    ckpt = None
    for step in range(10000):
        @ray.remote(max_calls=1)
        def f(ckpt):
            trainer = PPOTrainer(config=config)
            if ckpt:
               trainer.restore(ckpt)
            info = trainer.train()
            print(pretty_print(info))
            ckpt = trainer.save()
            trainer.stop()
            del trainer
            gc.collect()
            return ckpt, info
        ckpt, results = ray.get(f.remote(ckpt))
        tune.report(**results)
        gc.collect()

def tune_run():
    ray.init(ignore_reinit_error=True, lru_evict=True)
    config = get_trainer_config(train_policies=['player_1', 'player_2'], env_config={})
    return tune.run(run_ppo, config=config, verbose=True, name="dummy_run", num_samples=1)


if __name__ == '__main__':
    tune_run()
