from rps_rllib import RPSNoise
import numpy as np

def test_create_env():
    env = RPSNoise({'noise_dim': 4})
    
def test_episode():
    env = RPSNoise({'noise_dim': 4})
    obs = env.reset()
    
    total_rew = np.zeros(2)
    
    for _ in range(10):
        a1 = np.random.randint(0, 2)
        a2 = np.random.randint(0, 2)
        obs, rew, done, info = env.step({'player1': a1, 'player2': a2})
        total_rew += [rew['player1'], rew['player2']]
    print(total_rew)
    assert np.sum(total_rew) == 0