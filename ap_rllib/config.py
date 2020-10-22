from copy import deepcopy
from functools import partial

import numpy as np
from ray import tune
from ray.rllib.agents.es import ESTrainer, ESTFPolicy
from ray.rllib.agents.ppo import APPOTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.appo_tf_policy import AsyncPPOTFPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune.schedulers import ASHAScheduler

from frankenstein.remote_trainer import ExternalTrainer
from gym_compete_rllib import create_env
from ap_rllib.helpers import sample_int, tune_int
from ray.tune.logger import pretty_print
from ap_rllib.bursts import bursts_config_increase, bursts_config

CONFIGS = {}


def register_config(name):
    """Register configuration."""
    def register_inner(f):
        global CONFIGS
        CONFIGS[name] = f()
        return f
    return register_inner


def build_trainer_config(config):
    """Obtain rllib config from tune config (populate additional fields)."""
    # determining environment parameters
    env_fcn = config['_env_fcn']
    env = env_fcn(config['_env'])
    obs_space, act_space, n_policies = env.observation_space, env.action_space, env.n_policies
    env.close()

    policies = config['_get_policies'](config=config, n_policies=n_policies, obs_space=obs_space, act_space=act_space)
    select_policy = config['_select_policy']

    config = deepcopy(config)
    config['_all_policies'] = sorted(policies.keys())

    if config['_update_withpolicies'] and '_iteration' in config:
        config = config['_update_withpolicies'](config, iteration=config['_iteration'])

    config1 = deepcopy(config)
    config1['multiagent'] = {}
    config1['multiagent']['policies'] = policies

    for k in config['_train_policies']:
        assert k in policies.keys(), f"Unknown policy {k} [range {policies.keys()}]"

    rl_config = {
        "env": config['_env_name_rllib'],
        "env_config": config['_env'],
        "multiagent": {
            "policies_to_train": config['_train_policies'],
            "policies": policies,
            "policy_mapping_fn": partial(select_policy, config=config1),
        },
        'tf_session_args': {'intra_op_parallelism_threads': config['_num_workers_tf'],
                            'inter_op_parallelism_threads': config['_num_workers_tf'],
                            'gpu_options': {'allow_growth': True},
                            'log_device_placement': True,
                            'device_count': {'CPU': config['_num_workers_tf']},
                            'allow_soft_placement': True
                            },
        "local_tf_session_args": {
            "intra_op_parallelism_threads": config['_num_workers_tf'],
            "inter_op_parallelism_threads": config['_num_workers_tf'],
        },
    }

    # filling in the rest of variables
    for k, v in config.items():
        if k.startswith('_'): continue
        rl_config[k] = v

    if config['_verbose']:
        print("Config:")
        print(pretty_print(rl_config))

    if config['_trainer'] == 'External' and '_tmp_dir' in config:
        rl_config['tmp_dir'] = config['_tmp_dir']

    return rl_config


def get_agent_config(agent_id, which, obs_space, act_space, config):
    """Get config for agent models (pretrained/from scratch/from scratch stable baselines."""
    agent_config_pretrained = (POLICIES[config['_trainer']], obs_space, act_space, {
        'model': {
            "custom_model": "GymCompetePretrainedModel",
            "custom_model_config": {
                "agent_id": agent_id - 1,
                "env_name": config['_env']['env_name'],
                "model_config": {},
                "name": "model_%s" % (agent_id - 1),
                "load_weights": True,
            },
        },

        "framework": config['framework'],
    })

    agent_config_from_scratch_sb = (POLICIES[config['_trainer']], obs_space, act_space, {
        'model': {
            "custom_model": "GymCompetePretrainedModel",
            "custom_model_config": {
                "agent_id": agent_id - 1,
                "env_name": config['_env']['env_name'],
                "model_config": {},
                "name": "model_%s" % (agent_id - 1),
                "load_weights": 'normalization_only',
            },
        },

        "framework": config['framework'],
    })

    agent_config_from_scratch = (POLICIES[config['_trainer']], obs_space, act_space, {
        "model": {
            **config['_model_params']
        },
        "framework": config['framework'],
        "observation_filter": "MeanStdFilter",
    })

    configs = {"pretrained": agent_config_pretrained,
               "from_scratch": agent_config_from_scratch,
               "from_scratch_sb": agent_config_from_scratch_sb}

    return configs[which]


def get_policies_default(config, n_policies, obs_space, act_space, policy_template="player_%d"):
    """Get the default policy dictionary."""
    policies = {policy_template % i: get_agent_config(agent_id=i, which=config['_policies'][i],
                                                      config=config,
                                                      obs_space=obs_space, act_space=act_space)
                for i in range(1, 1 + n_policies)}
    return policies


def select_policy_default(agent_id, config):
    """Select policy at execution."""
    agent_ids = ["player_1", "player_2"]
    return agent_id


def get_default_config():
    """Default configuration for YouShallNotPass."""
    config = {}

    config["kl_coeff"] = 1.0
    config["_num_workers_tf"] = 4
    config["use_gae"] = True
    config["num_gpus"] = 0

    config["_env_name_rllib"] = "multicomp"
    config["_env_fcn"] = create_env
    config['_policies'] = [None, "from_scratch", "pretrained"]
    config["_env"] = {'with_video': False,
                      "SingleAgentToMultiAgent": False,
                      "env_name": "multicomp/YouShallNotPassHumans-v0"}
    config['framework'] = 'tfe'

    config['_train_policies'] = ['player_1']
    config['_call'] = {}
    config['_trainer'] = "PPO"
    config['_policy'] = "PPO"
    config['_call']['checkpoint_freq'] = 0
    config['_call']['name'] = 'adversarial_youshallnotpass'
    config['_train_steps'] = 99999999
    config['_update_config'] = None
    config['_run_inline'] = False
    config['_postprocess'] = None

    config['num_envs_per_worker'] = 4
    config['_log_error'] = True
    config['_model_params'] = {
        "use_lstm": False,
        "fcnet_hiddens": [64, 64],
        # "custom_action_dist": "DiagGaussian",
        "fcnet_activation": "tanh",
        "free_log_std": True,
    }

    config['_select_policy'] = select_policy_default
    config['_get_policies'] = get_policies_default
    config['_do_not_train_policies'] = []
    config['_update_withpolicies'] = None

    return config


def update_config_external_template(config):
    """Set trainer to external."""

    # best parameters from the paper
    config['train_batch_size'] = 16384
    config['lr'] = 3e-4
    config['sgd_minibatch_size'] = 4096
    config['num_sgd_iter'] = 4
    config['rollout_fragment_length'] = 100

    # run ID to communicate to the http trainer
    config['run_uid'] = '_setme'

    # stable baselines accepts full episodes
    config["batch_mode"] = "complete_episodes"

    # stable baselines server address
    config["http_remote_port"] = "http://127.0.0.1:50001"

    # no gpus, stable baselines might use them
    config['num_gpus'] = 0

    # set trainer class
    config['_trainer'] = "External"
    config['_policy'] = "PPO"

    # tuned
    config['num_envs_per_worker'] = 10
    config['num_workers'] = 3
    return config


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
    config['_call']['name'] = "adversarial_tune_fine"
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
    config['_call']['name'] = "adversarial_tune_fine2"
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
    config['_call']['name'] = "adversarial_best"
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
    config['_call']['name'] = "adversarial_linear"
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
    config['_call']['name'] = "adversarial_sizes"
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
    config['_call']['name'] = "adversarial_es"
    config['_call']['num_samples'] = 1
    config['_trainer'] = 'ES'

    # config['_run_inline'] = True
    config['_call']['stop'] = {'timesteps_total': 100000000}  # 30 million time-steps']
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers'] + 1}}

    return config


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
    config['_call']['name'] = "adversarial_external_test_sb"
    config['_call']['num_samples'] = 2

    config['train_batch_size'] = 1024
    config['lr'] = 3e-4
    config['sgd_minibatch_size'] = 1024
    config['num_sgd_iter'] = 2
    config['rollout_fragment_length'] = 100
    config['num_workers'] = 0
    config['num_envs_per_worker'] = 2
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

    config['_call']['name'] = "cartpole_external_sb"
    config['_call']['num_samples'] = 1
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
    config['_call']['name'] = "adversarial_test"
    config['_call']['num_samples'] = 1

    # config['_run_inline'] = True

    return config


@register_config(name='sample_speed')
def get_config_sample_speed():
    """Search for best num_workers/num_envs configuration."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = 16384
    config['lr'] = 3e-4
    config['sgd_minibatch_size'] = 4096
    config['num_sgd_iter'] = 4
    config['rollout_fragment_length'] = 128
    config['num_workers'] = tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    config['num_envs_per_worker'] = tune.grid_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config['num_gpus'] = 0
    config['_train_steps'] = 20
    config["batch_mode"] = "complete_episodes"

    config['_trainer'] = "PPO"
    config['_policy'] = "PPO"
    config['_call']['name'] = "adversarial_speed"
    config['_call']['num_samples'] = 1
    config['_call']['resources_per_trial'] = {
        "custom_resources": {"tune_cpu": tune.sample_from(lambda spec: spec.config.num_workers + 1)}}  # upper bound

    # config['_run_inline'] = True

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
    config['_call']['name'] = "adversarial_tune_bursts_exp_sb"
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
    config['_call']['name'] = "adversarial_tune_recover"
    config['_call']['num_samples'] = 4
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
    config['_call']['name'] = "adversarial_tune_recover_sb"
    config['_call']['num_samples'] = 4
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers']}}
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
    config['_call']['name'] = "adversarial_tune_bursts"
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
    config['_call']['name'] = "adversarial_tune_bursts_exp"
    config['_call']['num_samples'] = 20
    return config


def get_policies_all(config, n_policies, obs_space, act_space, policy_template="player_%d%s"):
    """Get a policy dictionary, both pretrained/from scratch."""
    which_arr = {"pretrained": "_pretrained", "from_scratch": ""}
    policies = {policy_template % (i, which_v): get_agent_config(agent_id=i, which=which_k, config=config,
                                                                 obs_space=obs_space, act_space=act_space)
                for i in range(1, 1 + n_policies)
                for which_k, which_v in which_arr.items()
                }
    return policies


def get_policies_withnormal_sb(config, n_policies, obs_space, act_space, policy_template="player_%d%s"):
    """Get a policy dictionary, both pretrained normal and adversarial opponents."""
    which_arr = {1:
                     {"from_scratch_sb": "_pretrained_adversary_sb",
                      "pretrained": "_pretrained_sb",
                      },
                 2:
                     {"pretrained": "_pretrained_sb"}
                 }
    policies = {policy_template % (i, which_v): get_agent_config(agent_id=i, which=which_k, config=config,
                                                                 obs_space=obs_space, act_space=act_space)
                for i in range(1, 1 + n_policies)
                for which_k, which_v in which_arr[i].items()
                }
    if config['_verbose']:
        print("Policies")
        print(policies.keys())
    return policies


def select_policy_opp_normal_and_adv_sb(agent_id, config, do_print=False):
    """Select policy at execution, normal-adversarial opponents."""
    p_normal = config['_p_normal']
    if agent_id == "player_1":
        out = np.random.choice(["player_1_pretrained_sb", "player_1_pretrained_adversary_sb"],
                               p=[p_normal, 1 - p_normal])
        if do_print or config['_verbose']:
            print('Chosen', out)
        return out
    elif agent_id == "player_2":
        # pretrained victim
        return "player_2_pretrained_sb"


def get_policies_pbt(config, n_policies, obs_space, act_space, policy_template="player_%d%s",
                     from_scratch_name="from_scratch"):
    """Get a policy dictionary, population-based training."""
    n_adversaries = config['_n_adversaries']
    which_arr = {1:
                     {"pretrained": ["_pretrained"],
                      from_scratch_name: ["_from_scratch_%03d" % i for i in range(1, n_adversaries + 1)]},
                 2: {"pretrained": ["_pretrained"]}
                 }

    policies = {
        policy_template % (i, which_v): get_agent_config(agent_id=i, which=which_k, config=config, obs_space=obs_space,
                                                         act_space=act_space)
        for i in range(1, 1 + n_policies)
        for which_k, which_v_ in which_arr[i].items()
        for which_v in which_v_
    }
    return policies


def select_policy_opp_normal_and_adv_pbt(agent_id, config, do_print=False):
    """Select policy at execution, PBT."""
    p_normal = config['_p_normal']
    n_adversaries = config['_n_adversaries']

    if agent_id == "player_1":
        out = np.random.choice(
            ["player_1_pretrained"] + ["player_1_from_scratch_%03d" % i for i in range(1, n_adversaries + 1)],
            p=[p_normal] + [(1 - p_normal) / n_adversaries for _ in range(n_adversaries)])
    elif agent_id == "player_2":
        # pretrained victim
        out = "player_2_pretrained"
    if do_print or config['_verbose']:
        print(f"Choosing {out} for {agent_id}")
    return out


def select_policy_opp_normal_and_adv(agent_id, config, do_print=False):
    """Select policy at execution, normal-adversarial opponents."""
    p_normal = config['_p_normal']
    if agent_id == "player_1":
        out = np.random.choice(["player_1_pretrained", "player_1"],
                               p=[p_normal, 1 - p_normal])
        if do_print or config['_verbose']:
            print('Chosen', out)
        return out
    elif agent_id == "player_2":
        # pretrained victim
        return "player_2_pretrained"


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
    config['_call']['name'] = "adversarial_tune_recover_withnormal_sb"
    config['_call']['num_samples'] = 5

    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers']}}
    config['_select_policy'] = select_policy_opp_normal_and_adv_sb
    config['_get_policies'] = get_policies_withnormal_sb
    config['_p_normal'] = 0.5
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
    config['_call']['name'] = "adversarial_tune_bursts_exp_withnormal"
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
    config['_call']['name'] = "adversarial_tune_bursts_exp_withnormal_pbt"
    config['_call']['num_samples'] = 100
    # ['humanoid_blocker', 'humanoid'],

    # config['_run_inline'] = True
    config['_select_policy'] = select_policy_opp_normal_and_adv_pbt
    config['_get_policies'] = get_policies_pbt
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
    config['_call']['name'] = "adversarial_tune_bursts_exp_withnormal_1adv_sb"
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
    config['_eval_steps'] = 1500
    config['_burst_exponent'] = tune.loguniform(1, 2.2, 2)
    config['_p_normal'] = tune.uniform(0.1, 0.9)
    config['_n_adversaries'] = 5  # tune_int(tune.uniform(1, 10))
    config['entropy_coeff'] = tune.uniform(0, 0.02)

    steps = (config['_train_steps'] + config['_eval_steps']) * config['train_batch_size']

    config['_call']['stop'] = {'timesteps_total': steps}
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers'] + 1}}
    config['_call']['name'] = "adversarial_tune_bursts_exp_withnormal_pbt_sb"
    config['_call']['num_samples'] = 50
    # ['humanoid_blocker', 'humanoid'],

    # config['_run_inline'] = True
    config['_select_policy'] = select_policy_opp_normal_and_adv_pbt
    config['_get_policies'] = partial(get_policies_pbt, from_scratch_name="from_scratch_sb")
    config['_do_not_train_policies'] = ['player_1_pretrained']
    return config


def get_trainer(config):
    """Get trainer from config."""
    # creating rllib config
    rl_config = build_trainer_config(config=config)
    return TRAINERS[config['_trainer']](config=rl_config)


TRAINERS = {'PPO': PPOTrainer,
            'APPO': APPOTrainer,
            'ES': ESTrainer,
            'External': ExternalTrainer}
POLICIES = {'PPO': PPOTFPolicy,
            'APPO': AsyncPPOTFPolicy,
            'ES': ESTFPolicy,
            'External': PPOTFPolicy}
