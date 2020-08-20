from gym.envs.registration import register
from functools import partial
import ray
import numpy as np
from multiprocessing import Queue

env_name = 'multicomp/YouShallNotPassHumans-v0'

from gym_compete_rllib.gym_compete_to_rllib import create_env, MultiAgentToSingleAgent

# issue: performance is not great -- because the env is not vectorized
# a better solution is to merge policies

@ray.remote
class act(object):
    def __init__(self):
        import tensorflow as tf
        tf.compat.v1.enable_eager_execution()
        from gym_compete_rllib.load_gym_compete_policy import get_policy_value_nets
        from gym_compete_rllib.gym_compete_to_rllib import model_to_callable

        self.policy_model_1 = model_to_callable(get_policy_value_nets(env_name, 1)['policy'])
        
    def predict(self, x):
        return self.policy_model_1(x)
    

#info = ray.init(ignore_reinit_error=True)

# actors = Queue()
# for _ in range(3):
#     actors.put(act.remote())

    
def create_env_embed_agent(env_name, remote_agent=True):
    """Create a single-agent env."""
    if remote_agent == True:
        #ray.init(ignore_reinit_error=True, address=info['redis_address'])
        ray.init(address='auto', redis_password='5241590000000000', ignore_reinit_error=True)
        actor = act.remote()
#        actor = actors.get()
        policy_model_1 = lambda o, actor=actor: ray.get(actor.predict.remote(o))
    elif remote_agent == 'dummy':
        policy_model_1 = lambda o: np.zeros(17)
    else:
        from gym_compete_rllib.load_gym_compete_policy import get_policy_value_nets
        from gym_compete_rllib.gym_compete_to_rllib import model_to_callable
        import tensorflow as tf
        tf.compat.v1.enable_eager_execution()
        policy_model_1 = model_to_callable(get_policy_value_nets(env_name, 1)['policy'])

    env = create_env(config=dict(with_video=False, env_name=env_name))
    env = MultiAgentToSingleAgent(env_config=dict(env=env, policies={'player_2': policy_model_1}))
    return env

env_name = 'multicomp/YouShallNotPassHumans-v0'
register(id='YouShallNotPassHumans-ZooV1-v0', entry_point=partial(create_env_embed_agent, env_name=env_name))