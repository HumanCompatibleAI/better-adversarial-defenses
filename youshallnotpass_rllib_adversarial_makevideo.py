import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from youshallnotpass_rllib_adversarial import build_trainer, build_trainer_config, get_config, created_envs, ray_init
import pickle, json, codecs
import argparse
from tqdm import tqdm

ray_init(shutdown=False)

parser = argparse.ArgumentParser(description='Produce a video from a checkpoint.')
parser.add_argument('--checkpoint', type=str,
                    help='Checkpoint json file')
parser.add_argument('--steps', type=int, default=10,
                    help='Evaluation steps')
parser.add_argument('--load_normal', type=bool, default=False,
                    help='Load normal opponent instead of the checkpoint')
args = parser.parse_args()


config = get_config()
config['train_policies'] = []
config['train_steps'] = args.steps
config['train_batch_size'] = 256
config['sgd_minibatch_size'] = 256
config['num_sgd_iter'] = 1
config['lr'] = 0

print("Load normal %s" % args.load_normal)

restore_state = None
env_config = {'with_video': True}
rl_config = build_trainer_config(train_policies=config['train_policies'],
                              config=config, num_workers=0, env_config=env_config, use_gpu=False, load_normal=args.load_normal)
print("Config", rl_config)
trainer = build_trainer(restore_state=None, config=rl_config)

if args.checkpoint:
  ck = json.load(open(args.checkpoint, 'r'))
  w = codecs.decode(ck['weights'].encode(), 'base64')
  trainer.set_weights(pickle.loads(w))
for _ in tqdm(range(config['train_steps'])):
    print(trainer.train()['hist_stats']['policy_player_1_reward'])
print("Your video is in")
print(created_envs[-1].video_recorder.path)
