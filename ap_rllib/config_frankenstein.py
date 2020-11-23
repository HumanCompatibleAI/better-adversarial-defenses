from ray import tune
from functools import partial
from ap_rllib.config_helpers import register_config, get_default_config, update_config_external_template, select_config, get_config_by_name
from ap_rllib.bursts import bursts_config, bursts_config_increase
from ap_rllib.config_pbt_helpers import select_policy_opp_normal_and_adv_pbt, get_policies_pbt, select_policy_opp_normal_and_adv_sb
from ap_rllib.config_pbt_helpers import get_policies_withnormal_sb
import os
from ap_rllib.ask_checkpoints import get_checkpoint_list, DEFAULT_PATH


@register_config(name='external_test')
def get_config_test_external():
    """Run with training via stable baselines."""
    # try changing learning rate
    config = get_default_config()
    config = update_config_external_template(config)

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1', 'player_2']
    config['_policies'] = [None, "from_scratch_sb", "pretrained"]
    config['_train_steps'] = 1
    config['_call']['num_samples'] = 2

    config['train_batch_size'] = 1024
    config['lr'] = 3e-4
    config['sgd_minibatch_size'] = 1024
    config['num_sgd_iter'] = 2
    config['rollout_fragment_length'] = 100
    config['num_workers'] = 0
    config['num_envs_per_worker'] = 2
    return config

@register_config(name='test_burst')
def get_config_test_bursts():
    """Run with bursts (small test run)."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = 4096
    config['lr'] = 1e-4
    config['sgd_minibatch_size'] = 4096
    config['num_sgd_iter'] = 2
    config['rollout_fragment_length'] = 1500
    config['num_workers'] = 10

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config['_update_config'] = bursts_config
    config['_train_steps'] = 1000000
    config['_burst_size'] = 2
    return config

@register_config(name='bursts_exp_sb')
def get_config_bursts_exp_sb():
    """Run with bursts (small test run)."""
    # try changing learning rate
    config = get_default_config()
    config = update_config_external_template(config)

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config['_update_config'] = bursts_config_increase
    config['_train_steps'] = 5000
    config['_eval_steps'] = 1500
    config['_burst_exponent'] = tune.loguniform(1.1, 2, 2)
    config['_policies'] = [None, "from_scratch_sb", "pretrained"]

    steps = (config['_train_steps'] + config['_eval_steps']) * config['train_batch_size']

    config['_call']['stop'] = {'timesteps_total': steps}
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers']}}
    config['_call']['num_samples'] = 10
    return config

@register_config(name='victim_recover_sb')
def get_config_victim_recover_sb():
    """Victim recovers from a pre-trained adversary."""
    # try changing learning rate
    config = get_default_config()
    config = update_config_external_template(config)

    config['_checkpoint_restore'] = './results/checkpoint-adv-external-3273'

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_2']
    config['_policies'] = [None, "from_scratch_sb", "pretrained"]
    config['_train_steps'] = 9999999999

    config['_call']['stop'] = {'timesteps_total': 50000000}  # 30 million time-steps']
    config['_call']['num_samples'] = 4
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers']}}
    return config

@register_config(name='bursts_exp_withnormal_1adv_sb')
def get_config_bursts_normal_1adv_sb():
    """One trial with bursts and PBT, 1 adversary."""
    # try changing learning rate
    config = get_default_config()
    config = update_config_external_template(config)

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = []
    config['_update_withpolicies'] = bursts_config_increase
    config['_train_steps'] = 10000
    config['_eval_steps'] = 1500
    config['_burst_exponent'] = tune.loguniform(1, 2.2, 2)
    config['_p_normal'] = 0.5  # tune.uniform(0.1, 0.9)
    config['_n_adversaries'] = 1  # tune_int(tune.uniform(1, 10))
    # config['entropy_coeff'] = tune.uniform(0, 0.02)

    steps = (config['_train_steps'] + config['_eval_steps']) * config['train_batch_size']

    config['_call']['stop'] = {'timesteps_total': steps}
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers']}}
    config['_call']['num_samples'] = 50
    # ['humanoid_blocker', 'humanoid'],

    # config['_run_inline'] = True
    config['_select_policy'] = select_policy_opp_normal_and_adv_pbt
    config['_get_policies'] = partial(get_policies_pbt, from_scratch_name="from_scratch_sb")
    config['_do_not_train_policies'] = ['player_1_pretrained']
    return config


@register_config(name='bursts_exp_withnormal_pbt_sb')
def get_config_bursts_normal_pbt_sb():
    """One trial with bursts and PBT."""
    # try changing learning rate
    config = get_default_config()
    config = update_config_external_template(config)

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = []
    config['_update_withpolicies'] = bursts_config_increase
    config['_train_steps'] = 10000
    config['_eval_steps'] = 2500
    config['_burst_exponent'] = tune.loguniform(1, 2.2, 2)
    config['_p_normal'] = 0.5 # tune.uniform(0.1, 0.9)
    config['_n_adversaries'] = 5 # tune_int(tune.uniform(1, 10))
    config['entropy_coeff'] = tune.uniform(0, 0.02)

    steps = (config['_train_steps'] + config['_eval_steps']) * config['train_batch_size']

    config['_call']['stop'] = {'timesteps_total': steps}
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers'] + 3}}
    config['_call']['num_samples'] = 100
    # ['humanoid_blocker', 'humanoid'],

    # config['_run_inline'] = True
    config['_select_policy'] = select_policy_opp_normal_and_adv_pbt
    config['_get_policies'] = partial(get_policies_pbt, from_scratch_name="from_scratch_sb")
    config['_do_not_train_policies'] = ['player_1_pretrained']
    return config

@register_config(name='defense_eval_interactive_sb', online=True)
def get_config_defense_eval_sb():
    config = get_default_config()
    config = update_config_external_template(config)

    config['_foreign_config'] = select_config(title="Defense training config")
    conf_name = get_config_by_name(config['_foreign_config'])['_call']['name']
    config['_restore_only'] = [('player_2_pretrained', 'player_2')]
    config['_checkpoint_list'] = get_checkpoint_list(path=os.path.join(DEFAULT_PATH, conf_name), ask_path=False)
    config['_checkpoint_restore'] = tune.grid_search(config['_checkpoint_list'])

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config['_policies'] = [None, "from_scratch_sb", "pretrained"]
    config['_train_steps'] = 9999999999

    config['_call']['stop'] = {'timesteps_total': 50000000}  # 30 million time-steps']
    config['_call']['num_samples'] = 2
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers']}}
    return config

@register_config(name='victim_recover_withnormal_sb')
def get_config_victim_recover_withnormal_sb():
    config = get_default_config()
    config = update_config_external_template(config)

    config['_checkpoint_restore_policy'] = {
        'player_1_pretrained_adversary_sb': './results/checkpoint-adv-external-3273-player_1.pkl'}

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_2_pretrained_sb']
    config['_train_steps'] = 9999999999

    config['_call']['stop'] = {'timesteps_total': 50000000}  # 30 million time-steps']
    config['_call']['num_samples'] = 5

    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers']}}
    config['_select_policy'] = select_policy_opp_normal_and_adv_sb
    config['_get_policies'] = get_policies_withnormal_sb
    config['_p_normal'] = 0.5
    return config