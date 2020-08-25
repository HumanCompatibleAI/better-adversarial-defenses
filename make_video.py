import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from gym_compete_rllib.gym_compete_to_rllib import created_envs
from train import build_trainer_config, ray_init
from config import PPOTrainer, get_config_test, get_config_linear
import pickle, json, codecs
import argparse
from tqdm import tqdm
import numpy as np

ray_init(shutdown=False)

parser = argparse.ArgumentParser(description='Produce a video from a checkpoint.')
parser.add_argument('--checkpoint', type=str,
                    help='Checkpoint file', required=True)
parser.add_argument('--steps', type=int, default=10,
                    help='Evaluation steps')
parser.add_argument('--load_normal', type=bool, default=False,
                    help='Load normal opponent instead of the checkpoint')
parser.add_argument('--no_video', type=bool, default=False,
                    help='Do the video?')

args = parser.parse_args()

config = get_config_linear()
config['_train_policies'] = []
config['_train_steps'] = args.steps
config['train_batch_size'] = 256
config['sgd_minibatch_size'] = 256
config['num_sgd_iter'] = 1
config['rollout_fragment_length'] = 200
config['lr'] = 0

print("Args:", args)

config['_env']['with_video'] = not args.no_video
config['_load_normal'] = args.load_normal

config['num_gpus'] = 0

num_workers = 0 if not args.no_video else 5

config['num_workers'] = num_workers

rl_config = build_trainer_config(config=config)
print("Config", rl_config)
trainer = PPOTrainer(config=rl_config)
trainer.restore(args.checkpoint)

stats = []

for _ in tqdm(range(config['_train_steps'])):
    stats = trainer.train()['hist_stats']['policy_player_1_reward']
    print(stats)
    
stats = np.array(stats)
trials = len(stats)
wins = 100. * np.sum(stats > 0) / trials
losses = 100. * np.sum(stats < 0) / trials
ties = 100. * np.sum(stats == 0) / trials
print(f"Total trials {trials} win rate {wins}%% loss rate {losses}%% tie rate {ties}%%" % ())


if not args.no_video:
    print("Your video is in")
    print(created_envs[-1].video_recorder.path)
