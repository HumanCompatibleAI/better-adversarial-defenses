import os
from copy import deepcopy
from functools import partial

import logging
import numpy as np
from dialog import Dialog
from ray import tune
from ray.rllib.agents.es import ESTrainer, ESTFPolicy
from ray.rllib.agents.ppo import APPOTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.appo_tf_policy import AsyncPPOTFPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune.logger import pretty_print
from ray.tune.schedulers import ASHAScheduler
from ray.tune.sample import Domain

from ap_rllib.bursts import bursts_config_increase, bursts_config
from ap_rllib.helpers import sample_int, tune_int
from frankenstein.remote_trainer import ExternalTrainer
from gym_compete_rllib import create_env
import inspect
from ap_rllib.config_info_callbacks import InfoCallbacks

# map config name -> config dict (for normal configs) or function (for 'online')
CONFIGS = {}

# map config name -> config description (or None)
CONFIG_DESCR = {}


# RLLib trainers
TRAINERS = {'PPO': PPOTrainer,
            'APPO': APPOTrainer,
            'ES': ESTrainer,
            'External': ExternalTrainer}

# RLLib policy map
POLICIES = {'PPO': PPOTFPolicy,
            'APPO': AsyncPPOTFPolicy,
            'ES': ESTFPolicy,
            'External': PPOTFPolicy}

def _get_config_dict():
    """Get the CONFIGS dictionary."""
    return CONFIGS

def get_config_names():
    """Get the names of all configs."""
    return sorted(CONFIGS.keys())

def get_config_attributes(name):
    """Get config properties such as online."""
    if name not in CONFIGS:
        raise ValueError(f"Wrong config name: {name}, possible names: {CONFIGS.keys()}")
    
    config = CONFIGS[name]
    result = {'name': name,
              'online': callable(config)
             }
    
    assert isinstance(config, dict) or callable(config), TypeError(f"Wrong config {name}: {type(CONFIGS[name])}")
    
    return result

def get_trainer(config):
    """Get trainer from config."""
    # creating rllib config
    rl_config = build_trainer_config(config=config)
    return TRAINERS[config['_trainer']](config=rl_config)


def get_config_by_name(name):
    """Get configuration for training by name."""
    attrs = get_config_attributes(name)
    if attrs['online']:
        config = CONFIGS[name]()
    else:
        config = CONFIGS[name]
        
    # setting the Tune Run name attribute
    config['_call']['name'] = name
    
    return config


def select_config(title=None):
    """Get config name (ask the user)."""
    d = Dialog()
    choices = []
    for c_key in sorted(CONFIGS.keys()):
        attrs = get_config_attributes(c_key)
        descr = str(CONFIG_DESCR[c_key])
        if attrs['online']:
            descr += " (interactive)"
        choices.append((c_key, descr))
    code, tag = d.menu("Select configuration:", choices=choices, width=100, title=title)
    assert code == 'ok', f"Invalid response: {code} {tag}"
    return tag


def register_config(name, online=False, descr=None):
    """Register configuration."""
    def register_inner(f, descr=descr, online=online):
        global CONFIGS, CONFIG_DESCR
        CONFIGS[name] = f if online else f()
        
        if descr is None:
            caller = inspect.currentframe().f_back
            config_module = str(caller.f_globals['__name__']).split('.')[-1]
            descr = config_module + '/' + str(f.__doc__)
        
        CONFIG_DESCR[name] = descr
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

    if config.get('_verbose', True):
        print("Config:")
        print(pretty_print(rl_config))

    if config['_trainer'] == 'External' and '_tmp_dir' in config:
        rl_config['tmp_dir'] = config['_tmp_dir']
        
    for key, val in rl_config.items():
        if isinstance(val, Domain):
            sampled_val = val.sample()
            rl_config[key] = sampled_val
            logging.warning(f"Trainer got a ray.tune.sample for parameter {key}: {type(val)}({val}). Replacing it with a sampled value {sampled_val}")

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
    config['callbacks'] = InfoCallbacks

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