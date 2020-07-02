#!/usr/bin/env python
# coding: utf-8

# In[1]:


from rps_rl import RPSEnv, RPSAgent, Monitor, Universe
import rock_paper_scissors as rps
from sacred import Experiment
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from copy import deepcopy
from tqdm.notebook import tqdm
import tensorflow as tf
import numpy as np
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.compat.v1.enable_eager_execution()

ex = Experiment()


@ex.config
def config():
    noise_dim = 4
    # how many agents to create?
    N_AGENTS = 10

    # how many games to play during test?
    N_GAMES = 1000


# environment
env = RPSEnv(noise_dim=noise_dim)

# to track the performance
m = Monitor(agents=AGENTS)


# # Threat estimation with training adversarial agent-models
# 1. There is an unknown opponent
# 2. We play with it a bit
# 3. Then, we train an opponent model which would have done same actions
# 4. We train ourselves against this model
# 5. We try ourselves against the original unknown opponent


A1 = RPSAgent(noise_dim=env.noise_dim, identity=0)
A2 = RPSAgent(noise_dim=env.noise_dim, identity=1)


A1.do_train = False
A2.do_train = False


# In[16]:


# to track the performance
m_estimate = Monitor(agents=AGENTS + [A1])

U = Universe(environment=env, agents=[A1, A2], monitor=m_estimate)
for _ in tqdm(range(50)):
    rew = U.episode()


# In[17]:


# creating a model of the aversary (so far, it is random)
adversary_agent_model = RPSAgent(noise_dim=env.noise_dim, identity=-1)


# In[18]:


# collecting initial data
xis = []
acts = []

for (_, _, (xi, a1, a2, (r1, r2))) in m_estimate.data:
    xis.append(xi)
    acts.append(a2)


# In[19]:


xis = np.array(xis)
acts = np.array(acts)


# In[20]:


def fit_rl_agent_on_supervised_data(
        agent, xis, acts, epochs=10, do_plot=False):
    """Fit the policy on given data in a supervised way."""
    optimizer = tf.keras.optimizers.Adam(1e-2)
    loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy()

    def epoch():
        """One supervised epoch."""

        # not working if optimizing with 2 optimizers

        with tf.GradientTape() as tape:
            ys = agent.model(xis)
            loss_superv = loss_fcn(acts, ys)
            loss_rl = agent.trainer.train(return_loss=True)
            if loss_rl is None:
                loss_rl = tf.Variable(0.0)
            loss_rl *= 0  # 30
            #print(loss_superv, loss_rl)
            loss = loss_superv + loss_rl
        grads = tape.gradient(loss, agent.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, agent.model.trainable_variables))

        return loss_superv.numpy(), loss_rl.numpy(), loss.numpy()

    losses = [epoch() for _ in range(epochs)]

    if do_plot:
        losses = np.array(losses)
        plt.title("Supervised loss")
        plt.plot(losses[:, 0], label="supervised")
        plt.plot(losses[:, 1], label="rl")
        plt.plot(losses[:, 2], label="total")
        plt.legend()
        plt.xlabel("Epoch")

    return losses


# Try annealing mixing coefficient.
#
# Or adding a constant "maximal distance to the data", and run supervised
# separate optimizer until reaching it.

# ## Bootstrapping adversary from existing data

# In[21]:


supervised_losses = []

# to track the performance
m_adv_train = Monitor(agents=AGENTS + [A1, adversary_agent_model])

# to allow it defeat US
A1.do_train = False
# training will be done with a modified loss
adversary_agent_model.do_train = False
U = Universe(
    environment=env,
    agents=[
        A1,
        adversary_agent_model],
    monitor=m_adv_train)


# In[22]:


for _ in range(10):

    plt.figure()

    for _ in tqdm(range(100)):
        rew = U.episode()

    # this is one supervised learning step
    supervised_losses += fit_rl_agent_on_supervised_data(
        adversary_agent_model, xis, acts, epochs=10)

    plt.subplot(1, 2, 1)
    plt.title("Supervised loss")
    supervised_losses = np.array(supervised_losses)
    plt.plot(supervised_losses[:, 0], label="supervised")
    plt.plot(supervised_losses[:, 1], label="rl")
    plt.plot(supervised_losses[:, 2], label="total")
    print(supervised_losses)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Reward")
    plt.plot(
        pd.Series(
            A1.reward_by_opponent[adversary_agent_model]).rolling(50).mean())
    plt.axhline(0)
    plt.axhline(1)
    plt.axhline(-1)

    plt.show()

    # resetting data
    adversary_agent_model.train_data = []


# In[ ]:


# ## now, the adversary is ready, and WE can train AGAINST it

# In[23]:


m_adv_train_main = Monitor(agents=AGENTS + [A1, adversary_agent_model])

# to allow it defeat US
A1.do_train = True
adversary_agent_model.do_train = False
U = Universe(
    environment=env,
    agents=[
        A1,
        adversary_agent_model],
    monitor=m_adv_train_main)


# In[24]:


for _ in tqdm(range(1000)):
    rew = U.episode()


# In[25]:


plt.title("Reward")
plt.plot(
    pd.Series(
        A1.reward_by_opponent[adversary_agent_model]).rolling(50).mean())
plt.axhline(0)
plt.axhline(1)
plt.axhline(-1)
plt.show()


# ## now we evaluate against the original opponent

# In[26]:


A1.do_train = False
A2.do_train = False
U = Universe(environment=env, agents=[A1, A2], monitor=m_estimate)
for _ in tqdm(range(100)):
    rew = U.episode()


# In[27]:


plt.title("Reward")
plt.plot(pd.Series(A1.reward_by_opponent[A2]).rolling(100).mean())
plt.axhline(0)
plt.axhline(1)
plt.axhline(-1)
plt.show()


# Success! We have trained an agent in simulation, and then defended against it!
#
# Effect size is small -> try GAIL -- there are CHAI/OpenAI implementations.
#
# With supervised, works better than with supervised + rl?
# RL goes into a specific direction?
