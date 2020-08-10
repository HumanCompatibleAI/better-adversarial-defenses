from ray import tune
from ray.tune.schedulers import ASHAScheduler
from gym_compete_rllib.gym_compete_to_rllib import create_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

def get_default_config():
    """Main config."""
    config = {}

    config["kl_coeff"] = 1.0
    config["_num_workers_tf"] = 32
    config["use_gae"] = True
    config["num_gpus"] = 0

    config["_env_name_rllib"] = "multicomp"
    config["_env_fcn"] = create_env
    config['_policies'] = [None, "from_scratch", "pretrained"]
    config["_env"] = {'with_video': False,
                     "env_name": "multicomp/YouShallNotPassHumans-v0"}
    config['framework'] = 'tfe'

    config['_train_policies'] = ['player_1']
    config['_policy_cls'] = PPOTFPolicy
    config['_call'] = {}
    config['_call']['trainer'] = PPOTrainer
    config['_call']['checkpoint_freq'] = 0
    config['_num_workers_tf'] = 32

    return config

def get_config_coarse():
    """Search in wide range."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = tune.loguniform(2048, 320000, 2)
    config['lr'] = tune.loguniform(1e-5, 1e-2, 10)
    config['sgd_minibatch_size'] = tune.loguniform(512, 65536, 2)
    config['num_sgd_iter'] = tune.uniform(1, 30)
    config['train_steps'] = 99999999
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


def get_config_fine():
    """Search in a smaller range."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = tune.loguniform(2048, 150000, 2)
    config['lr'] = tune.loguniform(1e-5, 1e-3, 10)
    config['sgd_minibatch_size'] = tune.loguniform(1000, 65536, 2)
    config['num_sgd_iter'] = tune.uniform(3, 30)
    config['train_steps'] = 99999999
    config['rollout_fragment_length'] = tune.loguniform(2000, 5000, 2)
    config['num_workers'] = 4
    config['use_gpu'] = False

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    config["batch_mode"] = "complete_episodes"
    config['_call']['name'] = "adversarial_tune_fine",
    config['_call']['num_samples'] = 300
    return config


def get_config_test():
    """One trial."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = 128
    config['lr'] = 1e-4
    config['sgd_minibatch_size'] = 128
    config['num_sgd_iter'] = 2
    config['train_steps'] = 99999999
    config['rollout_fragment_length'] = 200
    config['num_workers'] = 4
    config['use_gpu'] = False

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    return config


CONFIGS = {'test': get_config_test(),
           'coarse': get_config_coarse(),
           'fine': get_config_fine()}


def get_agent_config(agent_id, which, obs_space, act_space, config):
    agent_config_pretrained = (config['_policy_cls'], obs_space, act_space, {
        'model': {
            "custom_model": "GymCompetePretrainedModel",
            "custom_model_config": {
                "agent_id": agent_id - 1,
                "env_name": config['_env']['env_name'],
                "model_config": {},
                "name": "model_%s" % (agent_id - 1)
            },
        },

        "framework": config['framework'],
    })

    agent_config_from_scratch = (config['_policy_cls'], obs_space, act_space, {
        "model": {
            "use_lstm": False,
            "fcnet_hiddens": [64, 64],
            # "custom_action_dist": "DiagGaussian",
            "fcnet_activation": "tanh",
            "free_log_std": True,
        },
        "framework": config['framework'],
        "observation_filter": "MeanStdFilter",
    })

    configs = {"pretrained": agent_config_pretrained,
               "from_scratch": agent_config_from_scratch}

    return configs[which]