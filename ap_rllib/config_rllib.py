from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ap_rllib.helpers import sample_int, tune_int
from ap_rllib.config_helpers import register_config, get_default_config, update_config_external_template
from ap_rllib.bursts import bursts_config, bursts_config_increase
from ap_rllib.config_pbt_helpers import select_policy_opp_normal_and_adv_pbt, get_policies_pbt, select_policy_opp_normal_and_adv_sb
from ap_rllib.config_pbt_helpers import get_policies_withnormal_sb, select_policy_opp_normal_and_adv, get_policies_all


@register_config(name='coarse')
def get_config_coarse():
    """Search hyperparams in a wide range."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = tune.loguniform(2048, 320000, 2)
    config['lr'] = tune.loguniform(1e-5, 1e-2, 10)
    config['sgd_minibatch_size'] = tune.loguniform(512, 65536, 2)
    config['num_sgd_iter'] = tune.uniform(1, 30)
    config['rollout_fragment_length'] = tune.loguniform(200, 5000, 2)
    config['num_workers'] = 4

    # ['humanoid_blocker', 'humanoid'],
    config['train_policies'] = ['player_1']

    custom_scheduler = ASHAScheduler(
        metric='policy_reward_mean/player_1',
        mode="max",
        grace_period=1000000,
        reduction_factor=2,
        max_t=50000000,
        time_attr='timesteps_total',
    )

    config['_call']['scheduler'] = custom_scheduler
    config['_call']['stop'] = {'timesteps_total': 50000000}  # 30 million time-steps']
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers']}}
    return config


@register_config(name='fine')
def get_config_fine():
    """Search in a smaller range."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = sample_int(tune.loguniform(2048, 150000, 2))
    config['lr'] = tune.loguniform(1e-5, 1e-3, 10)
    config['sgd_minibatch_size'] = sample_int(tune.loguniform(1000, 65536, 2))
    config['num_sgd_iter'] = sample_int(tune.uniform(3, 30))
    config['rollout_fragment_length'] = sample_int(tune.loguniform(2000, 5000, 2))
    config['num_workers'] = 4

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config["batch_mode"] = "complete_episodes"
    config['_call']['num_samples'] = 300
    return config


@register_config(name='fine2')
def get_config_fine2():
    """Search in a smaller range, second set."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = sample_int(tune.loguniform(2048, 50000, 2))
    config['lr'] = tune.loguniform(1e-5, 1e-3, 10)
    config['sgd_minibatch_size'] = sample_int(tune.loguniform(1000, 25000, 2))
    config['num_sgd_iter'] = sample_int(tune.uniform(3, 30))
    config['rollout_fragment_length'] = sample_int(tune.loguniform(2000, 5000, 2))
    config['num_workers'] = 4

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config["batch_mode"] = "complete_episodes"
    config['_call']['num_samples'] = 300

    # config['_run_inline'] = True
    config['_call']['stop'] = {'timesteps_total': 50000000}  # 30 million time-steps']
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers'] + 1}}

    return config


@register_config(name='best')
def get_config_best():
    """Run with best hyperparams."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = 42880
    config['lr'] = 0.000755454
    config['sgd_minibatch_size'] = 22628
    config['num_sgd_iter'] = 5
    config['rollout_fragment_length'] = 2866
    config['num_workers'] = 4

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config["batch_mode"] = "complete_episodes"
    config['_call']['num_samples'] = 4

    # config['_run_inline'] = True
    config['_call']['stop'] = {'timesteps_total': 100000000}  # 30 million time-steps']
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers'] + 1}}

    return config


@register_config(name='linear')
def get_config_linear():
    """Trying the linear policy."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = 42880
    config['lr'] = 0.000755454
    config['sgd_minibatch_size'] = 22628
    config['num_sgd_iter'] = 5
    config['rollout_fragment_length'] = 2866
    config['num_workers'] = 8

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config["batch_mode"] = "complete_episodes"
    config['_call']['num_samples'] = 4
    # config['_run_inline'] = True

    config['_call']['stop'] = {'timesteps_total': 100000000}  # 30 million time-steps']
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers'] + 1}}
    config['_model_params'] = {
        "custom_model": "LinearModel",
        "custom_model_config": {
            "model_config": {},
            "name": "model_linear"
        },
    }

    return config


@register_config(name='sizes')
def get_config_sizes():
    """Trying different network sizes."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = 42880
    config['lr'] = 0.000755454
    config['sgd_minibatch_size'] = 22628
    config['num_sgd_iter'] = 5
    config['rollout_fragment_length'] = 2866
    config['num_workers'] = 4

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config["batch_mode"] = "complete_episodes"
    config['_call']['num_samples'] = 4
    # config['_run_inline'] = True

    config['_run_inline'] = True
    config['_call']['stop'] = {'timesteps_total': 100000000}  # 30 million time-steps']
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers'] + 3}}
    config['_model_params'] = {
        "use_lstm": False,
        "fcnet_hiddens": tune.grid_search([[256, 256, 256], [256, 256], [64, 64], [64, 64, 64]]),
        # "custom_action_dist": "DiagGaussian",
        "fcnet_activation": "tanh",
        "free_log_std": True,
    }

    return config


@register_config(name='test')
def get_config_test():
    """Do a test run."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = 16384
    config['lr'] = 3e-4
    config['sgd_minibatch_size'] = 4096
    config['num_sgd_iter'] = 4
    config['rollout_fragment_length'] = 128
    config['num_workers'] = 3

    config['num_envs_per_worker'] = 10

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config['num_gpus'] = 0
    config['_train_steps'] = 10000
    config["batch_mode"] = "complete_episodes"

    config['_trainer'] = "PPO"
    config['_policy'] = "PPO"
    config['_call']['num_samples'] = 1

    # config['_run_inline'] = True

    return config

@register_config(name='victim_recover')
def get_config_victim_recover():
    """Victim recovers from a pre-trained adversary."""
    # try changing learning rate
    config = get_default_config()

    config['_checkpoint_restore'] = './results/checkpoint-adv-67'

    config['train_batch_size'] = 42879
    config['lr'] = 0.000755454
    config['sgd_minibatch_size'] = 22627
    config['num_sgd_iter'] = 5
    config['rollout_fragment_length'] = 2865
    config['num_workers'] = 4

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_2']
    config['_train_steps'] = 9999999999

    config['_call']['stop'] = {'timesteps_total': 50000000}  # 30 million time-steps']
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers']}}
    config["batch_mode"] = "complete_episodes"
    config['_call']['num_samples'] = 4
    return config

@register_config(name='bursts_exp_withnormal')
def get_config_bursts_normal():
    """One trial with bursts + training against the normal opponent as well."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = 42879
    config['lr'] = 0.000755454
    config['sgd_minibatch_size'] = 22627
    config['num_sgd_iter'] = 5
    config['rollout_fragment_length'] = 2865
    config['num_workers'] = 4

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = []
    config['_update_config'] = bursts_config_increase
    config['_train_steps'] = 5000
    config['_eval_steps'] = 1500
    config['_burst_exponent'] = tune.loguniform(1.1, 2, 2)
    config['_p_normal'] = 0.5
    config['entropy_coeff'] = tune.uniform(0, 0.01)

    steps = (config['_train_steps'] + config['_eval_steps']) * config['train_batch_size']

    config['_call']['stop'] = {'timesteps_total': steps}
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers']}}
    config["batch_mode"] = "complete_episodes"
    config['_call']['num_samples'] = 1
    # ['humanoid_blocker', 'humanoid'],

    config['_select_policy'] = select_policy_opp_normal_and_adv
    config['_get_policies'] = get_policies_all
    return config


@register_config(name='bursts_exp_withnormal_pbt')
def get_config_bursts_normal_pbt():
    """One trial with bursts and PBT."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = 42879
    config['lr'] = 0.000755454
    config['sgd_minibatch_size'] = 22627
    config['num_sgd_iter'] = 5
    config['rollout_fragment_length'] = 2865
    config['num_workers'] = 4

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config['_update_config'] = bursts_config_increase
    config['_train_steps'] = 5000
    config['_eval_steps'] = 1500
    config['_burst_exponent'] = tune.loguniform(1.1, 2, 2)
    config['_p_normal'] = tune.uniform(0.1, 0.9)
    config['_n_adversaries'] = tune_int(tune.uniform(1, 10))
    config['entropy_coeff'] = tune.uniform(0, 0.01)

    steps = (config['_train_steps'] + config['_eval_steps']) * config['train_batch_size']

    config['_call']['stop'] = {'timesteps_total': steps}
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers']}}
    config["batch_mode"] = "complete_episodes"
    config['_call']['num_samples'] = 100
    # ['humanoid_blocker', 'humanoid'],

    # config['_run_inline'] = True
    config['_select_policy'] = select_policy_opp_normal_and_adv_pbt
    config['_get_policies'] = get_policies_pbt
    return config

@register_config(name='burst')
def get_config_bursts():
    """Grid search with bursts."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = 42879
    config['lr'] = 0.000755454
    config['sgd_minibatch_size'] = 22627
    config['num_sgd_iter'] = 5
    config['rollout_fragment_length'] = 2865
    config['num_workers'] = 4

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config['_update_config'] = bursts_config
    config['_train_steps'] = 100000000
    config['_burst_size'] = tune.grid_search([0, 1, 50, 200, 400, 800, 1600])  # loguniform(1, 500, 10)

    config['_call']['stop'] = {'timesteps_total': 100000000}  # 30 million time-steps']
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers']}}
    config["batch_mode"] = "complete_episodes"
    config['_call']['num_samples'] = 3
    return config


@register_config(name='bursts_exp')
def get_config_bursts_exp():
    """Hyperparam tuning with bursts."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = 42879
    config['lr'] = 0.000755454
    config['sgd_minibatch_size'] = 22627
    config['num_sgd_iter'] = 5
    config['rollout_fragment_length'] = 2865
    config['num_workers'] = 4

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config['_update_config'] = bursts_config_increase
    config['_train_steps'] = 5000
    config['_eval_steps'] = 1500
    config['_burst_exponent'] = tune.loguniform(1.1, 2, 2)

    steps = (config['_train_steps'] + config['_eval_steps']) * config['train_batch_size']

    config['_call']['stop'] = {'timesteps_total': steps}
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers']}}
    config["batch_mode"] = "complete_episodes"
    config['_call']['num_samples'] = 20
    return config


