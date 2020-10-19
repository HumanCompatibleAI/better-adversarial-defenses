from gym_compete_rllib import create_env
from ray.tune.registry import ENV_CREATOR, _global_registry


def test_create_env():
    env_creator = _global_registry.get(ENV_CREATOR, "multicomp")
    env_config = {'with_video': False,
                  "SingleAgentToMultiAgent": False,
                  "env_name": "multicomp/YouShallNotPassHumans-v0"}
    env = env_creator(env_config)

    assert env.n_policies == 2
    assert env.observation_space.shape == (380,)
    assert env.action_space.shape == (17,)
    assert env.player_names == ['player_1', 'player_2']

    def episode(env):
        obs = env.reset()
        def check_obs(obs, error_on_empty=True):
            assert isinstance(obs, dict)
            if error_on_empty:
                assert set(obs.keys()) == set(env.player_names), f"{obs.keys()} {env.player_names}"
            assert all([o.shape == env.observation_space.shape for o in obs.values()])
        check_obs(obs)
        while True:
            actions = {p: env.action_space.sample() for p in env.player_names}
            obs, reward, done, info = env.step(actions)
            check_obs(obs, error_on_empty=False)
            if done['__all__']:
                break

    for _ in range(10):
        episode(env)

if __name__ == '__main__':
    test_create_env()