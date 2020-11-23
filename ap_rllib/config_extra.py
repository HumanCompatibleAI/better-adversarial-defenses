from ap_rllib.config_helpers import register_config, get_default_config, update_config_external_template

@register_config(name='es')
def get_config_es():
    """Run with random search (evolutionary strategies)."""
    # try changing learning rate
    config = get_default_config()
    del config['kl_coeff']
    del config['use_gae']

    config['num_workers'] = 20

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config["batch_mode"] = "complete_episodes"
    config['_call']['num_samples'] = 1
    config['_trainer'] = 'ES'

    # config['_run_inline'] = True
    config['_call']['stop'] = {'timesteps_total': 100000000}  # 30 million time-steps']
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers'] + 1}}

    return config



@register_config(name='external_cartpole')
def get_config_cartpole_external():
    """Run with training via stable baselines."""
    # try changing learning rate
    config = get_default_config()
    config = update_config_external_template(config)

    config['train_batch_size'] = 4096
    config['lr'] = 3e-4
    config['sgd_minibatch_size'] = 1026
    config['num_sgd_iter'] = 4
    config['rollout_fragment_length'] = 100
    config['num_workers'] = 4
    config['num_envs_per_worker'] = 8

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config['_policies'] = [None, "from_scratch_sb"]
    config['_train_steps'] = 100
    config["_env"] = {'with_video': False,
                      "SingleAgentToMultiAgent": True,
                      "env_name": "InvertedPendulum-v2"}

    config['_call']['num_samples'] = 1
    return config

@register_config(name='test_appo')
def get_config_test_appo():
    """One trial APPO."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = 2048
    config['lr'] = 1e-4
    config['num_sgd_iter'] = 1
    # config['rollout_fragment_length'] = 128
    config['num_workers'] = 0

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1', 'player_2']
    config['num_gpus'] = 0

    config['_trainer'] = "APPO"

    config['_run_inline'] = True

    config['_train_steps'] = 10

    # config['num_envs_per_worker'] = 1
    return config