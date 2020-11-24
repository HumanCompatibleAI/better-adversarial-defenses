# Better defenses against Adversarial Policies in Reinforcement Learning
Defending against <a href="https://adversarialpolicies.github.io/">adversarial policies</a> in <a href="https://arxiv.org/pdf/1710.03748.pdf">YouShallNotPass</a> by running adversarial fine-tuning. Policies are trained in an alternating fashion: after training the adversary for t<sub>1</sub> steps, the victim is trained for t<sub>2</sub> steps, then the adversary is trained again for t<sub>3</sub> time-steps and so on. Training times t<sub>i</sub> increase exponentially. 

Bursts training: (left) training opponents ('normal' pre-trained, adversary trained from scratch, victim policy) in an alternating way (middle) 'burst' size (right) win rate<br />
<img height="200" src="https://github.com/HumanCompatibleAI/better-adversarial-defenses/blob/master/results/bursts_pbt_1adv/which_and_burst_size.png" /> <img height="200" src="https://github.com/HumanCompatibleAI/better-adversarial-defenses/blob/master/results/bursts_pbt_1adv/win_rate.png" />

Bursts training: (left) mean reward for agents, (right) value loss for agents<br />
<img height="300" src="https://github.com/HumanCompatibleAI/better-adversarial-defenses/blob/master/results/bursts_pbt_1adv/reward_mean.png" /> <img height="300" src="https://github.com/HumanCompatibleAI/better-adversarial-defenses/blob/master/results/bursts_pbt_1adv/value_loss.png" />

In this repository:
1. <a href="https://arxiv.org/pdf/1710.03748.pdf">`YouShallNotPass`</a> environment is exported into <a href="https://docs.ray.io/en/latest/rllib.html">rllib</a> as a [multiagent environment](gym_compete_rllib/__init__.py)
2. Training in 'bursts' is [implemented](ap_rllib/bursts.py#L55): victim or the adversary are trained against each other, the policy trained changes every t<sub>i</sub> time-steps, and t<sub>i</sub> increase exponentially
3. Victim is trained against [multiple adversaries](ap_rllib/config.py#L975) as well as the normal opponent ('population-based training')
4. [Stable Baselines](https://github.com/hill-a/stable-baselines) are [connected](./frankenstein) to rllib to train by sampling with rllib and optimizing with Stable Baslines

## Setup
[![Build Status](https://travis-ci.com/HumanCompatibleAI/better-adversarial-defenses.svg?branch=master)](https://travis-ci.com/HumanCompatibleAI/better-adversarial-defenses)

### Very simple: pull a [Docker](https://www.docker.com/) image
1. First, pull the image:

   `$ docker pull humancompatibleai/better-adversarial-defenses`

2. To run tests (will ask for a [MuJoCo](http://www.mujoco.org/) license)

   `$ docker run -it humancompatibleai/better-adversarial-defenses`

3. To run the terminal:

   `$ docker run -it humancompatibleai/better-adversarial-defenses /bin/bash`

### A bit harder: build a Docker image
<details><summary>Click to open</summary>
<p>
   
1. Install [Docker](https://www.docker.com/) and [git](https://git-scm.com/)
2. Clone the repository: `$ git clone https://github.com/HumanCompatibleAI/better-adversarial-defenses.git`
3. Build the Docker image: `$ docker build -t ap_rllib better-adversarial-defenses`
3. Run tests: `$ docker container run -it ap_rllib`
4. Run shell: `$ docker container run -it ap_rllib /bin/bash`
</p>
</details>



### Hard: set up the environment manually
<details><summary>Click to open</summary>
<p>

Assuming Ubuntu Linux distribution or a compatible one.

Tested in [Ubuntu](https://ubuntu.com/) 18.04.5 LTS and [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10). GPU is not required for the project.

Full installation can be found in [`Dockerfile`](Dockerfile).

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. `$ git clone --recursive https://github.com/HumanCompatibleAI/better-adversarial-defenses.git`
2. Create environments from files `adv-tf1.yml` and `adv-tf2.yml` (tf1 is used for stable baselines, and tf2 is used for rllib):
   * `$ conda env create -f adv-tf1.yml`
   * `$ conda env create -f adv-tf2.yml`
3. Install [MuJoCo](http://www.mujoco.org/) 1.13. On headless setups, install [Xvfb](https://en.wikipedia.org/wiki/Xvfb)
4. Install [MongoDB](https://www.mongodb.com/) and create a database `chai`
5. Install `gym_compete` and `aprl` via setup.py (included into the repository as submodules):
   * `$ pip install -e multiagent-competition`
   * `$ pip install -e pip install -e adversarial-policies`
6. Having [ray](https://github.com/ray-project/ray) 0.8.6 installed, run `$ python ray/python/ray/setup-dev.py` to patch your ray installation
7. Install fonts for rendering: `$ conda install -c conda-forge mscorefonts; mkdir ~/.fonts; cp $CONDA_PREFIX/fonts/*.ttf ~/.fonts; fc-cache -f -v`
8. Install the project: `$ pip install -e .`
</p>
</details>

## How to train
1. To test the setup with rllilb PPO trainer, run:

   `(adv-tf2) $ python -m ap_rllib.train --tune test`

   * The script will automatically log results to [Sacred](https://sacred.readthedocs.io/) and [Tune](https://docs.ray.io/en/latest/tune/index.html)
   * By-default, the script asks which configuration to run, but it can be set manually with the `--tune` argument.
   * Log files will appear in `~/ray_results/run_type/run_name`. Use [TensorBoard](https://www.tensorflow.org/tensorboard) in this folder.,
      - `run_type` is determined by the configuration (`config['_call']['name']` attribute). See [`config.py`](ap_rllib/config.py).
      - `run_name` is determined by [tune](https://docs.ray.io/en/latest/tune/index.html) -- see output of the train script.
   * Checkpoints will be in `~/ray_results/xxx/checkpoint_n/` where `xxx` and `n` are stored in the log files, one entry for every iteration. See an [example notebook](ap_rllib_experiment_analysis/youshallnotpass_rllib-analysis-external-recover-withnormal.ipynb) or [a script](ap_rllib_experiment_analysis/get_last_checkpoint.py) obtaining the last checkpoint for details on how to do that.

   * Some specifig configurations:
     - `--tune external_cartpole` runs training in InvertedPendulum, using Stable Baselines PPO implementation.
       * Before running, launch the Stable Baselines server `(adv-tf1) $ python -m frankenstein.stable_baselines_server`
          - By-default, each policy is trained in a separate thread, so that environment data collection resumes as soon as possible
          - However, this increases the number of threads significantly in case of PBT and many parallel tune trials.
          - If the number of threads is too high, the `--serial` option disables multi-threaded training in Stable Baselines Server
          - The overhead is not that significant, as training finishes extremely quickly compared to data collection
     - `--tune bursts_exp_withnormal_pbt_sb` will run training with Stable Baselines + Bursts + Normal opponent included + PBT (multiple adversaries)
   * `--verbose` enables some additional output
   * `--show_config` only shows configuration and exits
   * `--resume` will re-start trials if there are already trials in the results directory with this name
     - notebook [tune_pre_restart.ipynb](ap_rllib_experiment_analysis/notebooks/tune_pre_restart.ipynb) allows to convert ray 0.8.6 checkpoints to ray 1.0.1 checkpoints
   * If you want to quickly iterate with your config (use smaller batch size and no remote workers), pass an option to the trainer
   
      `--config_override='{"train_batch_size": 1000, "sgd_minibatch_size": 1000, "num_workers": 0, "_run_inline": 1}'`
   * Large number of processes might run into the open files limit. This might help: `ulimit -n 999999`

2. To make a video:

   * (only on headless setups): `$ Xvfb -screen 0 1024x768x24&; export DISPLAY=:0`

   * Run `(adv-tf2) $ python -m ap_rllib.make_video --checkpoint path/to/checkpoint/checkpoint-xxx --config your-config-at-training --display $DISPLAY`

     - `--steps n` number of steps to make (1 is `256`steps which is approximately 1 episode)
     - `--load_normal True` evaluate against normal opponent instead of the trained one
     - `--no_video True` will disable video. Use this to evaluate the performance with more episodes faster


## Design choices
1. We use ray because of its multi-agent support, and thus we have to use TensorFlow 2.0
2. We use stable baselines for training because we were unable to replicate results with rllib, even with an independent search for hyperparameters.
3. We checkpoint the ray trainer and restore it, and run the whole thing in a separate process to circumvent the <a href="https://github.com/ray-project/ray/issues/9964">ray memory leak issue</a>


## Files and folders structure
<details><summary>Click to open</summary>
<p>

Files:
* [`ap_rllib/train.py`](ap_rllib/train.py) the main train script
* [`ap_rllib/config.py`](ap_rllib/config.py) configurations for the train script
* [`ap_rllib/helpers.py`](ap_rllib/helpers.py) helper functions for the whole project
* [`ap_rllib/make_video.py`](ap_rllib/make_video.py) creates videos for the policies
* [`frankenstein/remote_trainer.py`](frankenstein/remote_trainer.py) implements an RLLib trainer that pickles data and sends the filename via HTTP
* [`frankenstein/stable_baselines_server.py`](frankenstein/stable_baselines_server.py) implements an HTTP server that waits for weights and samples, then trains the policy and returns the updated weights
* [`frankenstein/stable_baselines_external_data.py`](frankenstein/stable_baselines_external_data.py) implements the 'fake' Runner that allows for the training using Stable Baselines ppo2 algorithm on existing data
* [`gym_compete_rllib/gym_compete_to_rllib.py`](gym_compete_rllib/gym_compete_to_rllib.py) implements the adapter for the `multicomp` to `rllib` environments, and the `rllib` policy that loads pre-trained weights from `multicomp`
* [`gym_compete_rllib/load_gym_compete_policy.py`](gym_compete_rllib/load_gym_compete_policy.py) loads the `multicomp` weights into a keras policy
* [`gym_compete_rllib/layers.py`](gym_compete_rllib/layers.py) implements the observation/value function normalization code from `MlpPolicyValue` ([`multiagent-competition/gym_compete/policy.py`](https://github.com/HumanCompatibleAI/multiagent-competition/blob/72c342c4178cf189ea336a743f74e445faa6183a/gym_compete/policy.py))


Folders:
* [`ap_rllib_experiment_analysis`](ap_rllib_experiment_analysis) contains notebooks that analyze runs
* [`frankenstein`](frankenstein) contains the code for integrating Stable Baselines and RLLib
* [`gym_compete_rllib`](gym_compete_rllib) connects rllib to the `multicomp` environment

Submodules:
* [`adversarial-policies`](https://github.com/HumanCompatibleAI/adversarial-policies/tree/2ab9a717dd6a94f1314af526adbac1a59855048a) is the original project by <a href="https://www.gleave.me/">Adam Gleave</a>
* [`multiagent-competition`](https://github.com/HumanCompatibleAI/multiagent-competition) contains the environments used in the original project, as well as saved weights
* [`ray`](https://github.com/HumanCompatibleAI/ray/tree/342881dba62a0ba2de8b21c4367ec0a5d229f78e) is a copy of the `ray` repository with <a href="https://github.com/HumanCompatibleAI/ray/compare/releases/0.8.6...HumanCompatibleAI:adv">patches</a> to make the project work


### Additional files (see folder `other`)
* [`memory_profile`](other/memory_profile), `oom_dummy` contains files and data to analyze the memory leak
* [`rock_paper_scissors`](other/rock_paper_scissors) contain code with sketch implementations of ideas on Rock-Paper-Scissors game
* [`tf_agents_ysp.py`](other/tf_agents_ysp.py) implements training in `YouShallNotPass` with <a href="https://github.com/tensorflow/agents">tf-agents</a>
* [`rlpyt_run.py`](other/rlpyt_run.py) implements training in `YouShallNotPass` with <a href="https://github.com/astooke/rlpyt">rlpyt</a>
* [`rs.ipynb`](other/rs.ipynb) implements random search with a constant output policy in `YouShallNotPass`
* [`evolve.ipynb`](other/evolve.ipynb) and [`evolve.py`](other/evolve.ipynb) implement training in `YouShallNotPass` with <a href="https://github.com/CodeReclaimers/neat-python">neat-python</a>

</p>
</details>
