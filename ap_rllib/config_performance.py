from ray import tune
from ap_rllib.config_helpers import register_config, get_default_config, update_config_external_template

@register_config(name='sample_speed')
def get_config_sample_speed():
    """Search for best num_workers/num_envs configuration."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = 16384
    config['_policies'] = [None, "from_scratch_sb", "pretrained"]
    config['lr'] = 3e-4
    config['sgd_minibatch_size'] = 4096
    config['num_sgd_iter'] = 4
    config['rollout_fragment_length'] = 100
    config['num_workers'] = tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    config['num_envs_per_worker'] = tune.grid_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config['num_gpus'] = 0
    config['_train_steps'] = 20
    config["batch_mode"] = "complete_episodes"

    config['_trainer'] = "PPO"
    config['_policy'] = "PPO"
    config['_call']['num_samples'] = 1
    config['_call']['resources_per_trial'] = {
        "custom_resources": {"tune_cpu": tune.sample_from(lambda spec: spec.config.num_workers + 10)}}  # upper bound

    # config['_run_inline'] = True

    return config