from _unused.youshallnotpass_rllib_adversarial_td3 import build_trainer, build_trainer_config, get_config, created_envs
import pickle, json, codecs
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Produce a video from a checkpoint.')
parser.add_argument('--checkpoint', type=str,
                    help='Checkpoint json file')
parser.add_argument('--steps', type=int, default=10,
                    help='Evaluation steps')
args = parser.parse_args()


config = get_config()
config['train_policies'] = []
config['train_steps'] = args.steps
config['train_batch_size'] = 256
config['lr'] = 1e-3

restore_state = None
env_config = {'with_video': True}
rl_config = build_trainer_config(train_policies=config['train_policies'],
                              config=config, num_workers=0, env_config=env_config)
trainer = build_trainer(restore_state=None, config=rl_config)

ck = json.load(open(args.checkpoint, 'r'))
w = codecs.decode(ck['weights'].encode(), 'base64')
trainer.set_weights(pickle.loads(w))
for _ in tqdm(range(config['train_steps'])):
    print(trainer.train()['hist_stats']['policy_player_1_reward'])
print("Your video is in")
print(created_envs[-1].video_recorder.path)
