from gym.envs.registration import register
from functools import partial
import ray

ray.init()
env_name = 'multicomp/YouShallNotPassHumans-v0'


@ray.remote
class policy_actor(object):
    def __init__(self):
        from gym_compete_rllib.load_gym_compete_policy import get_policy_value_nets
        from gym_compete_rllib.gym_compete_to_rllib import model_to_callable
        self.c = model_to_callable(get_policy_value_nets(env_name, 1)['policy'])
        
    def call(self, *args):
        return self.c(*args)
    
    
#pool = ray.util.ActorPool([policy_actor.remote() for _ in range(10)])
p = policy_actor.remote()
    
def create_env_embed_agent(env_name):
    """Create a single-agent env."""
    print(dir())
    import sys
#     for k, v in list(sys.modules.items()):
#         if k.startswith('tensorflow'):
#             sys.modules.pop(k)
#         print(k)
    from gym_compete_rllib.gym_compete_to_rllib import create_env, MultiAgentToSingleAgent
    from gym_compete_rllib.load_gym_compete_policy import get_policy_value_nets
    from gym_compete_rllib.gym_compete_to_rllib import model_to_callable
    import gym
    
    
    env = create_env(config=dict(with_video=False, env_name=env_name))
    print("===Created environment...")
    
    #policy_model_1 = lambda o, pool=pool: list(pool.map(lambda a, v: a.call.remote(v), [o]))[0]
#     policy_model_1 = model_to_callable(get_policy_value_nets(env_name, 1)['policy'])
    policy_model_1 = lambda o, p=p: ray.get(p.call.remote(o))
    print("===Created policy...")
    env = MultiAgentToSingleAgent(env_config=dict(env=env, policies={'player_2': policy_model_1}))
    print("===Created MultiToSingle")
    return env

env_name = 'multicomp/YouShallNotPassHumans-v0'
register(id='YouShallNotPassHumans-ZooV1-v0', entry_point=partial(create_env_embed_agent, env_name=env_name))