# Better defenses against Adversarial Policies in Reinforcement Learning
Improving on the <a href="https://adversarialpolicies.github.io/">original publication</a>

In this repository:
1. `YouShallNotPass` environment is exported into rllib as a multiagent environment
2. Training in 'bursts' is implemented: victim or the adversary are trained against a frozen opponent for a number of steps
3. Victim is trained against multiple adversaries as well as the normal opponent ('population-based training')
4. Stable Baselines are connected to rllib to train by sampling with rllib and optimizing with Stable Baslines

Bursts training: (left) training opponents ('normal' pre-trained, adversary trained from scratch, victim policy) in an alternating way (middle) 'burst' size (right) win rate<br />
<img height="200" src="https://github.com/HumanCompatibleAI/better-adversarial-defenses/blob/master/results/bursts_pbt_1adv/which_and_burst_size.png" /> <img height="200" src="https://github.com/HumanCompatibleAI/better-adversarial-defenses/blob/master/results/bursts_pbt_1adv/win_rate.png" />

Bursts training: (left) mean reward for agents, (right) value loss for agents<br />
<img height="300" src="https://github.com/HumanCompatibleAI/better-adversarial-defenses/blob/master/results/bursts_pbt_1adv/reward_mean.png" /> <img height="300" src="https://github.com/HumanCompatibleAI/better-adversarial-defenses/blob/master/results/bursts_pbt_1adv/value_loss.png" />

## Setup
[![Build Status](https://travis-ci.com/HumanCompatibleAI/better-adversarial-defenses.svg?branch=master)](https://travis-ci.com/HumanCompatibleAI/better-adversarial-defenses)

### Using Docker
1. Install Docker and git
2. Clone the repository: `git clone https://github.com/HumanCompatibleAI/better-adversarial-defenses.git`
3. Build the Docker image: `docker build -t ap_rllib better-adversarial-defenses`
3. Run tests: `docker container run -it ap_rllib`
4. Run shell: `docker container run -it ap_rllib /bin/bash`

### Using Ubuntu
Assuming Ubuntu Linux distribution or a compatible one.

Tested in Ubuntu 18.04.5 LTS and WSL. GPU is not required for the project.

Full installation can be found in `Dockerfile`.

1. Install miniconda
2. `git clone --recursive https://github.com/HumanCompatibleAI/better-adversarial-defenses.git`
2. Create environments from files `adv-tf1.yml` and `adv-tf2.yml` (tf1 is used for stable baselines, and tf2 is used for rllib)
3. Install MuJoCo 1.13. On headless setups, install Xvfb
4. Install mongodb and create a database `chai`
5. Install `gym_compete` and `aprl` via setup.py (included into the repository as submodules)
6. Having ray 0.8.6 installed, run `python ray/python/ray/setup-dev.py` to patch your ray installation
7. Install fonts for rendering: `conda install -c conda-forge mscorefonts; mkdir ~/.fonts; cp $CONDA_PREFIX/fonts/*.ttf ~/.fonts; fc-cache -f -v`
8. Install the library: `pip install -e .`

## How to train
1. To test the setup with rllilb PPO trainer, run:

`(tf2) $ python -m ap_rllib.train --tune test`

Log files will appear in `~/ray_results/run_type/run_name`. Use tensorboard in this folder. Checkpoints will be in `~/ray_results/iteration_name` where `iteration_name` can be obtained from `run_type/run_name` in variable `checkpoint_rllib` (see notebooks in the analysis folder). `run_type` is a string corresponding to the `test` argument in the command that you ran. See CONFIGS variable in `config.py`.

2. To make a video:

(on headless setups): `$ Xvfb -screen 0 1024x768x24&; export DISPLAY=:0`

`(tf2) $ python -m ap_rllib.make_video --checkpoint path/to/checkpoint/checkpoint-xxx --config your-config-at-training --display $DISPLAY`

Options:
`--load_normal True` to evaluate against normal opponent instead of the trained one
`--steps n` number of steps to make (1 is `256`steps which is approximately 1 episode)
`--no_video True` will disable video. Use this to evaluate the performance with more episodes faster

3. To run PBT with bursts:
`(tf2) $ python -m ap_rllib.train --tune bursts_exp_withnormal_pbt`

4. To run training with stable baselines:
Running Stable Baselines server:

`(tf1) $ python -m frankenstein.stable_baselines_server`

5. Training in Inverted Pendulum (single-agent):

`(tf2) $ python -m ap_rllib.train --tune external_cartpole`

`(tf2) $ python -m ap_rllib.make_video --checkpoint path/checkpoint-xxx --display $DISPLAY --config external_cartpole --steps 1`


### Tips and tricks
* If you want to quickly iterate with your config (use smaller batch size and no remote workers), pass an option to the trainer:
`python -m ap_rllib.train ... ... ... --config_override='{"train_batch_size": 1000, "sgd_minibatch_size": 1000, "num_workers": 0, "_run_inline": 1}'`

* If you want to output additional information, add the following option:
`python -m ap_rllib.train ... ... ... --verbose`

* stable baselines server and the `train.py` scripts should be launched from the same folder, otherwise the temporary files will not be propagated

## Design choices
1. We use ray because of its multi-agent support, and thus we have to use TensorFlow 2.0
2. We use stable baselines for training because we were unable to replicate results with rllib, even with an independent search for hyperparameters.
3. We checkpoint the ray trainer and restore it, and run the whole thing in a separate process to circumvent the <a href="https://github.com/ray-project/ray/issues/9964">ray memory leak issue</a>


## Files and folders

Files:
* `ap_rllib/train.py` the main train script
* `ap_rllib/config.py` configurations for the train script
* `ap_rllib/helpers.py` helper functions for the whole project
* `ap_rllib/make_video.py` creates videos for the policies
* `frankenstein/remote_trainer.py` implements an RLLib trainer that pickles data and sends the filename via HTTP
* `frankenstein/stable_baselines_server.py` implements an HTTP server that waits for weights and samples, then trains the policy and returns the updated weights
* `frankenstein/stable_baselines_external_data.py` implements the 'fake' Runner that allows for the training using Stable Baselines ppo2 algorithm on existing data
* `gym_compete_rllib/gym_compete_to_rllib.py` implements the adapter for the `multicomp` to `rllib` environments, and the `rllib` policy that loads pre-trained weights from `multicomp`
* `gym_compete_rllib/load_gym_compete_policy.py` loads the `multicomp` weights into a keras policy
* `gym_compete_rllib/layers.py` implements the observation/value function normalization code from `MlpPolicyValue` (`multiagent-competition/gym_compete/policy.py`)


Folders:
* `ap_rllib_experiment_analysis` contains notebooks that analyze runs
* `frankenstein` contains the code for integrating Stable Baselines and RLLib
* `gym_compete_rllib` connects rllib to the `multicomp` environment

Submodules:
* `adversarial-policies` is the original project by <a href="https://www.gleave.me/">Adam Gleave</a>
* `multiagent-competition` contains the environments used in the original project, as well as saved weights
* `ray` is a copy of the `ray` repository with <a href="https://github.com/HumanCompatibleAI/ray/compare/releases/0.8.6...HumanCompatibleAI:adv">patches</a> to make the project work


### Additional files (see folder `other`)
* `memory_profile`, `oom_dummy` contains files and data to analyze the memory leak
* `rock_paper_scissors` contain code with sketch implementations of ideas on Rock-Paper-Scissors game
* `tf_agents_ysp.py` implements training in `YouShallNotPass` with <a href="https://github.com/tensorflow/agents">tf-agents</a>
* `rlpyt_run.py` implements training in `YouShallNotPass` with <a href="https://github.com/astooke/rlpyt">rlpyt</a>
* `rs.ipynb` implements random search with a constant output policy in `YouShallNotPass`
* `evolve.ipynb` and `evolve.py` implement training in `YouShallNotPass` with <a href="https://github.com/CodeReclaimers/neat-python">neat-python</a>
