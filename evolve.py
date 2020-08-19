import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from gym_compete_rllib.gym_compete_to_rllib import create_env, MultiAgentToSingleAgent, model_to_callable
from gym_compete_rllib.load_gym_compete_policy import get_policy_value_nets
from gym_compete_rllib.test_single_agent_env import episode
import gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import ray
from time import time
import neat
import pickle


env_name = 'multicomp/YouShallNotPassHumans-v0'
fc_config_filename = 'fc.config'
num_workers = 64
eval_episodes = 100
population_size = 50

ray.shutdown()
info = ray.init(ignore_reinit_error=True, log_to_driver=False)

env = create_env(config=dict(with_video=False, env_name=env_name))
policy_model_1 = model_to_callable(get_policy_value_nets(env_name, 1)['policy'])
env = MultiAgentToSingleAgent(env_config=dict(env=env, policies={'player_2': policy_model_1}))

@ray.remote
class Evaluator(object):
    """Evaluates a policy in the 1-agent env."""
    def __init__(self, env_name):
        env = create_env(config=dict(with_video=False, env_name=env_name))
        policy_model_1 = model_to_callable(get_policy_value_nets(env_name, 1)['policy'])
        self.env = MultiAgentToSingleAgent(env_config=dict(env=env, policies={'player_2': policy_model_1}))
    def episode(self, policy):
        return episode(self.env, policy)
    
def rewards(pool, ps):
    """Rewards for an array of policies."""
    return [v for v in pool.map(lambda a, v: a.episode.remote(v), ps)]

def compute_rewards(pool, p, total_episodes=100):
    """Compute rewards using a pool."""
    return [v for v in pool.map_unordered(lambda a, v: a.episode.remote(v), [p] * total_episodes)]

def evaluate_genomes(genomes, config):
    """Evaluate genomes using a pool"""
    nets = []
    for gid, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(lambda x, net=net: net.activate(x))
    policies = [ray.put(net) for net in nets]
    policies = [net for net in policies for _ in range(eval_episodes)]
    rs = rewards(pool, policies)
    rs = np.array(rs).reshape((len(genomes), eval_episodes))
    rs = np.mean(rs, axis=1)
    for (gid, g), r in zip(genomes, rs):
        g.fitness = r
    global stats, p
    pickle.dump([stats, p], open('evolve_result.pkl', 'wb'))
    return rs


actors = [Evaluator.remote(env_name=env_name) for _ in range(num_workers)]
pool = ray.util.ActorPool(actors)

config_initial = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, fc_config_filename)

config_initial.genome_config.num_inputs = env.observation_space.shape[0]
config_initial.genome_config.num_outputs = env.action_space.shape[0]
config_initial.pop_size = population_size

game_fc_config_filename = 'fc-1.config'

config_initial.save(game_fc_config_filename)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, game_fc_config_filename)

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(5))

winner = p.run(evaluate_genomes, 99999)

pickle.dump([stats, p], open('evolve_result.pkl', 'wb'))
