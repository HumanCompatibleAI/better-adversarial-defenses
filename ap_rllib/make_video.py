import argparse

parser = argparse.ArgumentParser(description='Produce a video from a checkpoint.')
parser.add_argument('--checkpoint', type=str,
                    help='Checkpoint file', required=True)
parser.add_argument('--config', type=str,
                    help='Config to load', default='test')
parser.add_argument('--steps', type=int, default=10,
                    help='Evaluation steps')
parser.add_argument('--load_normal', type=bool, default=False,
                    help='Load normal opponent instead of the checkpoint')
parser.add_argument('--no_video', type=bool, default=False,
                    help='Do the video?')
parser.add_argument('--display', type=str, default=':0',
                    help='Display to set')


def make_video(args):
    """Make video given arguments, return a dictionary with results."""

    # imports are done here to allow running this in multiple processes
    # without tensorflow multiprocessing issues
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from gym_compete_rllib import created_envs
    from ap_rllib.train import ray_init
    from ap_rllib.config import build_trainer_config, get_config_by_name
    from ap_rllib import config
    from ap_rllib.config_helpers import TRAINERS
    from tqdm import tqdm
    import numpy as np

    ray_init(shutdown=False)

    os.environ['DISPLAY'] = args.display

    config = get_config_by_name(args.config)

    # making config simpler
    config['_train_policies'] = []
    config['_train_steps'] = args.steps
    config['train_batch_size'] = 256
    config['sgd_minibatch_size'] = 256
    config['num_sgd_iter'] = 1
    config['rollout_fragment_length'] = 200
    config['lr'] = 0
    config['_verbose'] = False
    config['_env']['with_video'] = not args.no_video

    config['num_gpus'] = 0

    num_workers = 0# if not args.no_video else 5

    config['num_workers'] = num_workers
    config['num_envs_per_worker'] = 1 if not args.no_video else 8

    # which opponent to load?
    if args.load_normal:
        rl_config = build_trainer_config(config=config)
        print("Config", rl_config)
        trainer1 = TRAINERS[config['_trainer']](config=rl_config)
        trainer1.restore(args.checkpoint)

        config['_policies'] = [None, "pretrained", "pretrained"]
        rl_config = build_trainer_config(config=config)
        trainer = TRAINERS[config['_trainer']](config=rl_config)
        trainer.get_policy('player_2').set_weights(trainer1.get_policy('player_2').get_weights())

    else:
        rl_config = build_trainer_config(config=config)
        print("Config", rl_config)
        trainer = TRAINERS[config['_trainer']](config=rl_config)
        trainer.restore(args.checkpoint)

    stats_all = {}
    
    # processing win/lose rates
    results = {}
    
    # doing rollouts
    for _ in tqdm(range(config['_train_steps'])):
        stats = trainer.train()
        stats_all = {x: y for x, y in stats['hist_stats'].items() if x.endswith('reward')}
        v_contact = stats.get('custom_metrics', {}).get('contacts_mean', None)
        if v_contact is not None:
            results['contacts_mean'] = v_contact
        
    print(stats_all)

    for player, stats in stats_all.items():
        stats = np.array(stats)
        trials = len(stats)
        wins = 100. * np.sum(stats > 0) / trials
        losses = 100. * np.sum(stats < 0) / trials
        ties = 100. * np.sum(stats == 0) / trials
        print(f"Player {player} total trials {trials} win rate {wins}%% loss rate {losses}%% tie rate {ties}%%" % ())

        results['players'] = list(stats_all.keys())
        results['trials_' + player] = trials
        results['wins_' + player] = wins
        results['losses_' + player] = losses
        results['ties_' + player] = ties
        
    print(results)

    # adding videos as results
    if not args.no_video:
        print("Your video is in")
        recorder = created_envs[-1].video_recorder
        if hasattr(recorder, 'path'):
            v = recorder.path
            print(v)
        else:
            v = created_envs[-1].videos
            v = [x[0] for x in v]
            for x in v:
                print(x)
        results['video'] = v

    return results


if __name__ == '__main__':
    args = parser.parse_args()
    make_video(args)
