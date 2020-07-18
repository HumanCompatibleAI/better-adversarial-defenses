from youshallnotpass_rllib_adversarial import build_trainer, build_trainer_config, config, created_envs
import pickle, json, codecs
import argparse

parser = argparse.ArgumentParser(description='Produce a video from a checkpoint.')
parser.add_argument('checkpoint', type=str,
                    help='Checkpoint json file')
args = parser.parse_args()


config['train_policies'] = []
config['train_steps'] = 5
config['train_batch_size'] = 4096

restore_state = None
env_config = {'with_video': True}
rl_config = build_trainer_config(restore_state=restore_state,
                              train_policies=config['train_policies'],
                              config=config, num_workers=0, env_config=env_config)
trainer = build_trainer(restore_state=None, config=rl_config)

ck = json.load(open(args.checkpoint, 'r'))
w = codecs.decode(ck['weights'].encode(), 'base64')
trainer.set_weights(pickle.loads(w))
for _ in range(config['train_steps']):
    trainer.train()
print("Your video is in")
print(created_envs[-1].video_recorder.path)
