from gym_compete_to_rllib import create_env
from tqdm import tqdm
import numpy as np

config = {'with_video': False}#False}
env = create_env(config=config)
obs = env.reset()
for _ in tqdm(range(50000000)):
    action_dict = {'player_1': np.random.randn(17), 'player_2': np.random.randn(17)}
    obs, rew, done, info = env.step(action_dict)
    if done['__all__']:
        env.reset()
env.close()
