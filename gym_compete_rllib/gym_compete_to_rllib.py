from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from gym_compete_rllib.load_gym_compete_policy import get_policy_value_nets
from ray.tune.registry import register_env
import datetime, uuid
from gym_compete_rllib.layers import UnconnectedVariableLayer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gym
import numpy as np
import tensorflow as tf
import gym_compete
from ray.rllib.models import ModelCatalog

# scaler for reward output for players
REWARD_SCALER = 1. / 100


def dct_to_float32(d):
    """Convert dictionary values to float32."""
    return {x: np.array(y, dtype=np.float32) for x, y in d.items()}


def model_to_callable(m):
    """Convert a Keras model to a callable for a single object."""
    def f(x):
        return m(np.array([x])).numpy()[0]
    return f


class MultiAgentToSingleAgent(gym.Env):
    """Embed an agent into a multi-agent environment."""
    def __init__(self, env_config):
        self._env = env_config['env']
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.policies = env_config['policies']
        self.players = []
        self.previous_obs = {}
        
    @staticmethod
    def dict_single_value(d):
        """Get single value in a dict."""
        assert len(d) == 1, f"Must have only 1 key, got {d.keys()}"
        return list(d.values())[0]
    
    def other_player_value(self, d, check_len=True, default=None):
        """Get value for the remaining player."""
        if check_len and default is None:
            assert len(d) - len(self.policies) == 1, f"Must have one value remaining. {d}"
        for k, v in d.items():
            if k not in self.policies:
                return v
        return default
        
    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)
        
    def reset(self):
        obs = self._env.reset()
        self.players = list(obs.keys())
        self.previous_obs = obs
        return self.other_player_value(obs)
    
    def step(self, action):
        actions = {}
        for p in self.players:
            if p in self.policies:
                policy = self.policies[p]
                actions[p] = policy(self.previous_obs[p])
            else:
                actions[p] = action
                
        obss, rews, dones, infos = self._env.step(actions)
        if obss:
            self.previous_obs = obss
        else:
            obss = self.previous_obs
                
        return self.other_player_value(obss), self.other_player_value(rews, default=np.array(0, dtype=np.float32)),\
               self.other_player_value(dones, check_len=False), self.other_player_value(infos, check_len=False)


class GymCompeteToRLLibAdapter(MultiAgentEnv):
    """Takes gym_compete env and makes a multi-agent RLLib env."""

    def __init__(self, env_cls, player_names=None):
        env = env_cls()
        super(GymCompeteToRLLibAdapter, self).__init__()
        assert isinstance(env.action_space, gym.spaces.Tuple) and len(env.action_space) == 2
        assert isinstance(env.observation_space, gym.spaces.Tuple) and len(env.observation_space) == 2
        if player_names is None:
            player_names = ["player_%d" % i for i in range(1, 1 + len(env.action_space))]
        self.player_names = player_names
        self.n_policies = 2
        self._env = env

        self.reset_dones()

    def close(self):
        self._env.close()

    def reset_dones(self):
        self.dones = {name: False for name in self.player_names}

    def pack_array(self, array):
        """Given an array for players, return a dict player->value."""
        assert len(self.player_names) == len(array)
        return {p: o for p, o in zip(self.player_names, array)}

    def unpack_dict(self, dct, default_value=None):
        """Given a dict player->value return an array."""
        return [dct.get(p, default_value) for p in self.player_names]

    def reset(self):
        observations = self._env.reset()
        return dct_to_float32(self.pack_array(observations))

    def step(self, action_dict, reward_scaler=REWARD_SCALER):        
        default_action = np.zeros(self.observation_space.shape)
        a1a2 = self.unpack_dict(action_dict, default_action)
        o1o2, r1r2, done, i1i2 = self._env.step(a1a2)
        obs = self.pack_array(o1o2)
        rew = self.pack_array(np.array(r1r2) * reward_scaler)
        infos = self.pack_array([i1i2[0], i1i2[1]])
        dones = self.pack_array([i1i2[0]['agent_done'],
                                 i1i2[1]['agent_done']])

        # done only if everyone is done
        dones = {p: done for p in dones.keys()}
        dones['__all__'] = done

        # removing observations for already finished agents
        for p in self.player_names:
            if self.dones[p]:
                del obs[p]
                del rew[p]
                del infos[p]

        for p in self.player_names:
            self.dones[p] = dones[p]

        # for adversarial training
        if 'player_1' in rew:
            rew['player_1'] = -infos['player_2']['reward_remaining'] * reward_scaler
            
        rew = dct_to_float32(rew)
        obs = dct_to_float32(obs)
        
        return obs, rew, dones, infos

    @property
    def observation_space(self):
        # for one agent
        return self._env.observation_space[0]

    @property
    def action_space(self):
        # for one agent
        return self._env.action_space[0]

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)


class KerasModelModel(TFModelV2):
    """Create an RLLib policy from policy+value keras models."""

    def __init__(self, *args, policy_net=None, value_net=None, **kwargs):
        super(KerasModelModel, self).__init__(*args, **kwargs)
        self.policy_net = policy_net
        self.value_net = value_net
        self.register_variables(policy_net.variables + value_net.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        model_out = tf.cast(self.policy_net(obs), tf.float32)
        self._value_out = tf.cast(self.value_net(obs), tf.float32)[:, 0]
        self._value_out = self._value_out * REWARD_SCALER
        return model_out, state

    def value_function(self):
        return self._value_out


class GymCompetePretrainedModel(KerasModelModel):
    """Load a policy from gym_compete."""

    def __init__(self, *args, **kwargs):
        env_name = args[3]['custom_model_config']['env_name']
        agent_id = args[3]['custom_model_config']['agent_id']
        nets = get_policy_value_nets(env_name, agent_id)
        n_out = int(nets['policy_mean_logstd_flat'].output_shape[1])
        super(GymCompetePretrainedModel, self).__init__(*args, **kwargs,
                                                        policy_net=nets['policy_mean_logstd_flat'],
                                                        value_net=nets['value'])
class LinearModel(KerasModelModel):
    """Linear model."""

    def __init__(self, *args, **kwargs):

        x = tf.keras.Input(shape=(380,))
        y = tf.keras.layers.Dense(17, activation=None, use_bias=True)(x)
        model_policy_mean = y
        model_policy_mean = tf.keras.layers.Reshape((17, 1), name='reshape_mean')(model_policy_mean)
        model_policy_inp = x
        
        model_policy_std = UnconnectedVariableLayer(name='std', shape=(17,))(x)
        model_policy_std = tf.keras.layers.Reshape((17, 1), name='reshape_std')(model_policy_std)
        model_policy_mean_std_ = tf.keras.layers.Concatenate(axis=2)([model_policy_mean, model_policy_std])
        model_policy_mean_std_flat_ = tf.keras.layers.Flatten(data_format='channels_first')(model_policy_mean_std_)
        model_policy_mean_std_flat = tf.keras.Model(inputs=model_policy_inp, outputs=model_policy_mean_std_flat_)

        policy_net = model_policy_mean_std_flat
        x = tf.keras.Input(shape=(380,))
        y = tf.keras.layers.Dense(64, activation='tanh')(x)
        y = tf.keras.layers.Dense(64, activation='tanh')(y)
        y = tf.keras.layers.Dense(1, activation=None)(y)
        value_net = tf.keras.models.Model(inputs=x, outputs=y)

        super(LinearModel, self).__init__(*args, **kwargs,
                                                        policy_net=policy_net,
                                                        value_net=value_net)


ModelCatalog.register_custom_model("GymCompetePretrainedModel", GymCompetePretrainedModel)
ModelCatalog.register_custom_model("LinearModel", LinearModel)


def gym_compete_env_with_video(env_name, directory=None):
    """Record videos from gym_compete environments using aprl."""

    try:
        from aprl.envs.wrappers import VideoWrapper
        from aprl.visualize.annotated_gym_compete import AnnotatedGymCompete
        from aprl.score_agent import default_score_config
    except:
        pass

    # hacks to make it work with tf2
    import sys
    from unittest.mock import Mock
    sys.modules['stable_baselines'] = Mock()
    import tensorflow as tf
    tf.Session = Mock()

    if directory is None:
        directory = 'video-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + str(uuid.uuid1())

    from aprl.envs.wrappers import VideoWrapper
    from aprl.visualize.annotated_gym_compete import AnnotatedGymCompete
    from aprl.score_agent import default_score_config

    config = default_score_config()
    env = gym.make(env_name)

    # print(config)
    # config['video_params']['annotation_params']['font'] = '/home/sergei/.fonts/times'
    
    resolution = config['video_params']['annotation_params']['resolution']
    # print(resolution)
    # resolution = [480, 270]

    env = AnnotatedGymCompete(env=env, env_name=env_name, agent_a_type=config['agent_a_type'],
                              agent_b_type=config['agent_b_type'],
                              agent_a_path=config['agent_a_path'], agent_b_path=config['agent_b_path'],
                              mask_agent_index=config['mask_agent_index'],
                              resolution=resolution,
                              font=config['video_params']['annotation_params']['font'],
                              font_size=config['video_params']['annotation_params']['font_size'],
                              short_labels=config['video_params']['annotation_params']['short_labels'],
                              camera_config=config['video_params']['annotation_params']['camera_config']
                              )

    env = VideoWrapper(env=env, directory=directory)

    # sys.modules['stable_baselines'] = b
    # delattr(tf, 'Session')

    return env


created_envs = []


def create_env(config):
    if config['with_video']:
        env = gym_compete_env_with_video(config['env_name'])
    else:
        env = gym.make(config['env_name'])
    created_envs.append(env)
    return GymCompeteToRLLibAdapter(lambda: env)


register_env("multicomp", create_env)
