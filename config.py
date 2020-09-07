from ray import tune
from ray.tune.schedulers import ASHAScheduler
from gym_compete_rllib.gym_compete_to_rllib import create_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import APPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.appo_tf_policy import AsyncPPOTFPolicy
from ray.rllib.agents.es import ESTrainer, ESTFPolicy
from remote_trainer import ExternalTrainer
from copy import deepcopy
import numpy as np


def bursts_config(config, iteration):
    """Updates config to train with bursts."""
    config_new = deepcopy(config)

    pretrain_time = config['_train_steps'] // 2
    evaluation_time = config['_train_steps'] // 2
    burst_size = int(config['_burst_size'])

    #n_bursts = pretrain_time // (2 * burst_size)

    # print("Pretrain time: %d" % pretrain_time)
    # print("Evaluation time: %d" % evaluation_time)
    # print("Burst size", burst_size)
    # print("Number of bursts", n_bursts)
    # print("Total iterations (true)", n_bursts * burst_size * 2 + evaluation_time)

    train_policies = config_new['_train_policies']

    # pretraining stage
    if iteration < pretrain_time:
        if burst_size == 0:
            train_policies = ['player_1', 'player_2']
        elif burst_size < 0:
            raise ValueError("Wrong burst_size %s" % burst_size)
        else:
            current_burst = iteration // burst_size
            if current_burst % 2 == 0:
                train_policies = ['player_1']
            else:
                train_policies = ['player_2']
    else:
        train_policies = ['player_1']

    config_new['_train_policies'] = train_policies
    return config_new


def bursts_config_increase(config, iteration):
    """Updates config to train with bursts."""
    config_new = deepcopy(config)

    train_time = config['_train_steps']
    evaluation_time = config['_eval_steps']
    exponent = config['_burst_exponent']

    if train_time + evaluation_time < iteration:
        print(f"Iteration {iteration} too high")


    train_policies = config_new['_train_policies']
    info = {}

    # pretraining stage
    if iteration < train_time:
        bs_float, bs = 1.0, 1
        passed = 0
        while passed + 2 * bs < iteration + 1:
            passed += 2 * bs # 2 agents in total
            bs_float = bs_float * exponent
            bs = round(bs_float)

        # last burst size is ours
        delta = iteration - passed
        first_stage = delta < bs

        train_policies = ['player_1'] if first_stage else ['player_2']
        info['type'] = 'train'
        info['bs'] = bs
        info['bs_float'] = bs_float
        info['passed'] = passed
        info['delta'] = delta
    else:
        train_policies = ['player_1']
        info['type'] = 'eval'

    assert len(train_policies) == 1
    train_p = train_policies[0]

    if '_n_adversaries' in config:
        if train_p == "player_1":
            train_policies = ["player_1_from_scratch_%03d" % (i + 1) for i in range(config['_n_adversaries'])]
        elif train_p == "player_2":
            train_policies = ["player_2_pretrained"]

    config_new['_train_policies'] = train_policies
    config_new['_burst_info'] = info
    return config_new

def get_policies_default(config, n_policies, obs_space, act_space, policy_template="player_%d"):
    """Get a policy dictionary."""
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
    """Main config."""
    config = {}

    config["kl_coeff"] = 1.0
    config["_num_workers_tf"] = 4
    config["use_gae"] = True
    config["num_gpus"] = 0

    config["_env_name_rllib"] = "multicomp"
    config["_env_fcn"] = create_env
    config['_policies'] = [None, "from_scratch", "pretrained"]
    config["_env"] = {'with_video': False,
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

    return config

def get_config_coarse():
    """Search in wide range."""
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

def sample_int(obj):
    """Convert tune distribution to integer."""
    return tune.sample_from(lambda _: round(obj.func(_)))

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


def get_config_fine2():
    """Search in a smaller range."""
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
    
    #config['_run_inline'] = True
    config['_call']['stop'] = {'timesteps_total': 50000000}  # 30 million time-steps']
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers'] + 1}}

    return config


def get_config_best():
    """Search in a smaller range."""
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
    
    #config['_run_inline'] = True
    config['_call']['stop'] = {'timesteps_total': 100000000}  # 30 million time-steps']
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers'] + 1}}

    return config

def get_config_linear():
    """Search in a smaller range."""
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
    #config['_run_inline'] = True
    
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

def get_config_sizes():
    """Search in a smaller range."""
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
    #config['_run_inline'] = True
    
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

def get_config_es():
    """Run with random search."""
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
    
    #config['_run_inline'] = True
    config['_call']['stop'] = {'timesteps_total': 100000000}  # 30 million time-steps']
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers'] + 1}}

    return config

def get_config_test_external():
    """One trial."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = 16384
    config['lr'] = 3e-4
    config['sgd_minibatch_size'] = 4096
    config['num_sgd_iter'] = 4
    config['rollout_fragment_length'] = 100
    config['num_workers'] = 5
    
    config['num_envs_per_worker'] = 10

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1']
    
    config['_policies'] = [None, "from_scratch_sb", "pretrained"]
    config['run_uid'] = '_setme'
    config['num_gpus'] = 0

    config['_trainer'] = "External"
    config['_policy'] = "PPO"

    #config['_run_inline'] = True
    config["batch_mode"] = "complete_episodes"
    config["http_remote_port"] = "http://127.0.0.1:50001"

    config['_train_steps'] = 10000

    config['_call']['name'] = "adversarial_external_sb"
    config['_call']['num_samples'] = 2
    return config



def get_config_test():
    """One trial."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = 2048
    config['lr'] = 1e-4
    config['sgd_minibatch_size'] = 512
    config['num_sgd_iter'] = 1
    config['rollout_fragment_length'] = 128
    config['num_workers'] = 4
    
    config['num_envs_per_worker'] = 4

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1', 'player_2']
    config['num_gpus'] = 0

    config['_trainer'] = "PPO"
    config['_policy'] = "PPO"

    config['_run_inline'] = True

    config['_train_steps'] = 10
    return config


def get_config_test_appo():
    """One trial APPO."""
    # try changing learning rate
    config = get_default_config()

    config['train_batch_size'] = 2048
    config['lr'] = 1e-4
    config['num_sgd_iter'] = 1
    #config['rollout_fragment_length'] = 128
    config['num_workers'] = 0

    # ['humanoid_blocker', 'humanoid'],
    config['_train_policies'] = ['player_1', 'player_2']
    config['num_gpus'] = 0

    config['_trainer'] = "APPO"

    config['_run_inline'] = True

    config['_train_steps'] = 10
    
    #config['num_envs_per_worker'] = 1
    return config


def get_config_test_bursts():
    """One trial."""
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


def get_config_victim_recover():
    """Victim recovers from a trained adversary."""
    # try changing learning rate
    config = get_default_config()
    
    config['_checkpoint_restore'] = './checkpoint-adv-67'

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


def get_config_bursts():
    """One trial with bursts."""
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
    config['_burst_size'] = tune.grid_search([0, 1, 50, 200, 400, 800, 1600]) # loguniform(1, 500, 10)

    config['_call']['stop'] = {'timesteps_total': 100000000}  # 30 million time-steps']
    config['_call']['resources_per_trial'] = {"custom_resources": {"tune_cpu": config['num_workers']}}
    config["batch_mode"] = "complete_episodes"
    config['_call']['name'] = "adversarial_tune_bursts"
    config['_call']['num_samples'] = 3
    return config

def get_config_bursts_exp():
    """One trial with bursts."""
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
    """Get a policy dictionary."""
    which_arr = {"pretrained": "_pretrained", "from_scratch": ""}
    policies = {policy_template % (i, which_v): get_agent_config(agent_id=i, which=which_k, config=config, 
        obs_space=obs_space, act_space=act_space)
        for i in range(1, 1 + n_policies)
        for which_k, which_v in which_arr.items()
        }
    return policies

def get_policies_pbt(config, n_policies, obs_space, act_space, policy_template="player_%d%s"):
    """Get a policy dictionary."""
    n_adversaries = config['_n_adversaries']
    which_arr = {1:
                  {"pretrained": ["_pretrained"], "from_scratch": ["_from_scratch_%03d" % i for i in range(1, n_adversaries + 1)]},
                 2: {"pretrained": ["_pretrained"]}
                }
    
    policies = {policy_template % (i, which_v): get_agent_config(agent_id=i, which=which_k, config=config, obs_space=obs_space, act_space=act_space)
        for i in range(1, 1 + n_policies)
        for which_k, which_v_ in which_arr[i].items()
        for which_v in which_v_ 
    }
    #policies['default_policy'] = (None, obs_space, act_space, {})#get_agent_config(agent_id=1, which="pretrained", config=config, obs_space=obs_space, act_space=act_space)
    return policies

def select_policy_opp_normal_and_adv_pbt(agent_id, config, do_print=False):
    """Select policy at execution."""
    p_normal = config['_p_normal']
    n_adversaries = config['_n_adversaries']
    
    if agent_id == "player_1":
        out = np.random.choice(["player_1_pretrained"] + ["player_1_from_scratch_%03d" % i for i in range(1, n_adversaries + 1)],
                p=[p_normal] + [(1 - p_normal) / n_adversaries for _ in range(n_adversaries)])
    elif agent_id == "player_2":
        # pretrained victim
        out = "player_2_pretrained"
    if do_print:
        print(f"Choosing {out} for {agent_id}")
    return out

def select_policy_opp_normal_and_adv(agent_id, config, do_print=False):
    """Select policy at execution."""
    p_normal = config['_p_normal']
    if agent_id == "player_1":
        out = np.random.choice(["player_1_pretrained", "player_1"],
                p=[p_normal, 1 - p_normal])
        if do_print:
            print('Chosen', out)
        return out
    elif agent_id == "player_2":
        # pretrained victim
        return "player_2_pretrained"

def get_config_bursts_normal():
    """One trial with bursts and training against the normal opponent as well."""
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

    #config['_run_inline'] = True
    config['_select_policy'] = select_policy_opp_normal_and_adv
    config['_get_policies'] = get_policies_all
    return config


def tune_compose(obj, f):
    """Apply f after sampling from obj."""
    return tune.sample_from(lambda x: f(obj.func(x)))

def tune_int(obj):
    """Convert result to int after sampling from obj."""
    return tune_compose(obj, round)

def get_config_bursts_normal_pbt():
    """One trial with bursts and training against the normal opponent as well."""
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
    config['_call']['name'] = "adversarial_tune_bursts_exp_withnormal"
    config['_call']['num_samples'] = 1
    # ['humanoid_blocker', 'humanoid'],

    #config['_run_inline'] = True
    config['_select_policy'] = select_policy_opp_normal_and_adv_pbt
    config['_get_policies'] = get_policies_pbt
    return config


CONFIGS = {'test': get_config_test(),
           'coarse': get_config_coarse(),
           'fine': get_config_fine(),
           'test_burst': get_config_test_bursts(),
           'burst': get_config_bursts(),
           'test_appo': get_config_test_appo(),
           'victim_recover': get_config_victim_recover(),
           'fine2': get_config_fine2(),
           'best': get_config_best(),
           'es': get_config_es(),
           'linear': get_config_linear(),
           'sizes': get_config_sizes(),
           'bursts_exp': get_config_bursts_exp(),
           'bursts_exp_withnormal': get_config_bursts_normal(),
           'bursts_exp_withnormal_pbt': get_config_bursts_normal_pbt(),
           'external': get_config_test_external(),
          }

TRAINERS = {'PPO': PPOTrainer,
            'APPO': APPOTrainer,
            'ES': ESTrainer,
            'External': ExternalTrainer}
POLICIES = {'PPO': PPOTFPolicy,
            'APPO': AsyncPPOTFPolicy,
            'ES': ESTFPolicy,
            'External': PPOTFPolicy}


def get_agent_config(agent_id, which, obs_space, act_space, config):
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
                "load_weights": False,
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
