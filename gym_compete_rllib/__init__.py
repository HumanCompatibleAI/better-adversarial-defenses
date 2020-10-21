from gym_compete_rllib.gym_compete_to_rllib import gym_compete_env_with_video, SingleAgentToMultiAgent, GymCompeteToRLLibAdapter
from gym.wrappers import Monitor
import gym
from ray.tune.registry import register_env


created_envs = []

def create_env(config):
    if config['with_video']:
        if config['env_name'].startswith('multicomp'):
            env = gym_compete_env_with_video(config['env_name'])
        else:
            env = Monitor(gym.make(config['env_name']), directory='./', force=True)
    else:
        env = gym.make(config['env_name'])
    created_envs.append(env)
    if config['SingleAgentToMultiAgent']:
        env = SingleAgentToMultiAgent(env)
    if config['env_name'].startswith('multicomp'):
        env = GymCompeteToRLLibAdapter(lambda: env)
    return env


register_env("multicomp", create_env)
