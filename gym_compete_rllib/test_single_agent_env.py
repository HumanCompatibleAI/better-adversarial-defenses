from gym_compete_rllib.gym_compete_to_rllib import create_env, MultiAgentToSingleAgent, model_to_callable
from gym_compete_rllib.load_gym_compete_policy import get_policy_value_nets
import gym
import numpy as np


def episode(env, pi):
    """One episode."""
    R = 0
    obs = env.reset()
    done = False
    while not done:
        act = pi(obs)
        obs, rew, done, info = env.step(act)
        R += rew
    return R

def test_env():
    """Test that we can convert multi-agent to single-agent e"""
    env_name = 'multicomp/YouShallNotPassHumans-v0'
    env = create_env(config=dict(with_video=False, env_name=env_name))
    policy_model_1 = model_to_callable(get_policy_value_nets(env_name, 1)['policy'])
    env = MultiAgentToSingleAgent(env_config=dict(env=env, policies={'player_2': policy_model_1}))

    random_policy = lambda _: np.random.randn(17)
    policy_model_0 = model_to_callable(get_policy_value_nets(env_name, 0)['policy'])

    for _ in range(3):
        episode(env, random_policy)
    for _ in range(3):
        episode(env, policy_model_0)

if __name__ == '__maenv':
    test_env()