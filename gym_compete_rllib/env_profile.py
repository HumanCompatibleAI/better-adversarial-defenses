import gym
from tqdm import tqdm
import numpy as np
env_name = 'multicomp/YouShallNotPassHumans-v0'
env = gym.make(env_name)
obs = env.reset()
for _ in tqdm(range(10000)):
    act = (np.zeros(17), np.zeros(17))
    obs, rew, done, info = env.step(act)
    if done:
        env.reset()


from gym_compete_rllib.gym_compete_to_rllib import create_env
from tqdm import tqdm
import numpy as np

config = {'with_video': False}#False}
env = create_env(config=config)
obs = env.reset()
for _ in tqdm(range(10000)):
    action_dict = {'player_1': np.random.randn(17), 'player_2': np.random.randn(17)}
    obs, rew, done, info = env.step(action_dict)
    if done['__all__']:
        env.reset()
env.close()
