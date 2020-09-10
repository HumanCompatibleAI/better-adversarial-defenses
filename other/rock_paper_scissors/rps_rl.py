from rock_paper_scissors.rock_paper_scissors import rewards, n_act
from copy import deepcopy
import tensorflow as tf
import numpy as np
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.compat.v1.enable_eager_execution()


class Trainer(object):
    """REINFORCE trainer for the agent."""

    def __init__(self, agent):
        self.agent = agent
        self.optimizer = tf.keras.optimizers.Adam(1e-2)

    def train(self, return_loss=False):
        loss = 0
        if not self.agent.train_data:
            return

        # @tf.function
        def get_reinforce_loss(data, model):
            loss = 0.0
            for (xi, a, r, opponent) in data:
                loss -= r * tf.math.log(model(np.array([xi]))[0][a])
            loss /= len(data)
            return loss

        if return_loss:
            return get_reinforce_loss(self.agent.train_data, self.agent.model)

        with tf.GradientTape() as tape:
            loss = get_reinforce_loss(self.agent.train_data, self.agent.model)

        grads = tape.gradient(loss, self.agent.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.agent.model.trainable_variables))
        self.agent.train_data = []


class RPSAgent(object):
    """Rock Paper Scissors agent."""

    def __init__(self, noise_dim, identity=None, train_every=10):
        self.noise_dim = noise_dim
        self.identity = identity
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(self.noise_dim,), activation='linear'),
            tf.keras.layers.Dense(n_act, activation=None),
            tf.keras.layers.Softmax(),
        ])
        self.trainer = Trainer(agent=self)
        self.do_train = True

        self.data = []
        self.train_data = []

        self.reward_by_opponent = {}
        self.train_every = train_every

    def step(self, xi):
        p = self.model(np.array([xi]))[0].numpy()
        p = p / np.sum(p)
        return np.random.choice(range(n_act), p=p)

    def register_episode(self, h, opponent=None):
        self.data.append(deepcopy(list(h)) + [opponent])
        self.train_data.append(self.data[-1])
        if opponent not in self.reward_by_opponent:
            self.reward_by_opponent[opponent] = []
        self.reward_by_opponent[opponent].append(h[-1])
        self._train()

    def _train(self):
        if not self.do_train:
            return
        if len(self.data) % self.train_every == 0:
            self.trainer.train()

    def __repr__(self):
        opponents = np.unique([x[-1].identity for x in self.data])
        return "<Agent id=%d wisdom=%d opponents=%d>" % (
            self.identity, len(self.data), len(opponents))


class RPSEnv(object):
    """Rock paper scissors environment."""

    def __init__(self, noise_dim=10):
        self.noise_dim = noise_dim

    def reset(self):
        pass

    def step(self, a1, a2):
        R = rewards(a1, a2)
        assert np.sum(R) == 0
        return R

    def reset(self):
        return np.random.randn(self.noise_dim)

    def __repr__(self):
        return "<Env noise_dim=%d>" % self.noise_dim


class Universe(object):
    """Interaction between 2 agents in the environment."""

    def __init__(self, environment, agents, monitor, invert_reward_2=False):
        self.environment = environment
        self.monitor = monitor
        self.agents = agents
        self.invert_reward_2 = invert_reward_2
        assert len(self.agents) == 2

    def episode(self):
        xi = self.environment.reset()
        a1 = self.agents[0].step(xi)
        a2 = self.agents[1].step(xi)
        rews = self.environment.step(a1, a2)

        if self.invert_reward_2:
            rews = [rews[0], rews[0]]

        episode = (xi, a1, a2, rews)
        self.agents[0].register_episode(
            (xi, a1, rews[0]), opponent=self.agents[1])
        self.agents[1].register_episode(
            (xi, a2, rews[1]), opponent=self.agents[0])
        self.monitor.register(A1=self.agents[0],
                              A2=self.agents[1],
                              episode=episode)
        return rews

    def __repr__(self):
        return "<Universe\n  Environment=%s\n  Agents=%s\n>" % (
            self.environment, self.agents)


class Monitor(object):
    """Tracks agent's performance."""

    def __init__(self, agents):
        self.agents = agents
        self.data = []

    def register(self, A1, A2, episode):
        assert A1 in self.agents
        assert A2 in self.agents
        self.data.append([A1, A2, deepcopy(episode)])

    def stats(self):
        action_stats = {A: [0 for _ in range(n_act)] for A in self.agents}
        reward_stats = {A: {x: 0 for x in [-1, 0, 1]} for A in self.agents}

        for (A1, A2, (xi, a1, a2, (r1, r2))) in self.data:
            action_stats[A1][a1] += 1
            action_stats[A2][a2] += 1
            reward_stats[A1][r1] += 1
            reward_stats[A2][r2] += 1

        return {'reward': reward_stats,
                'action': action_stats}

    def __repr__(self):
        return "<Monitor games=%d>" % len(self.data)
