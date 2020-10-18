#!/bin/bash

# Disabling GPU
export CUDA_VISIBLE_DEVICES=-1

# launching mongodb
sudo service mongodb start

# running pytest
conda run -n adv-tf1 pytest -s -v gym_compete_rllib/test_load_gym_compete_policy.py::test_load
conda run -n adv-tf2 pytest -s -v gym_compete_rllib/test_load_gym_compete_policy.py::test_load
conda run -n adv-tf2 pytest -s -v ap_rllib
conda run -n adv-tf1 pytest -s -v gym_compete_rllib/test_load_gym_compete_policy.py::test_prediction
conda run -n adv-tf2 pytest -s -v gym_compete_rllib/test_load_rllib_env.py

# launching the stable baselines server
screen -Sdm "sb_server" conda run -n adv-tf1 python -m frankenstein.stable_baselines_server

# launching training with a few iterations
conda run -n adv-tf2 python -m ap_rllib.train --tune external_test

# obtaining the checkpoint
checkpoint=$(conda run -n adv-tf2 python -m ap_rllib_experiment_analysis.get_last_checkpoint --config external_test)

# making a video
conda run -n adv-tf2 python -m ap_rllib.make_video --config external_test --checkpoint $checkpoint --display $DISPLAY --steps 1

# closing the screen session
screen -S "sb_server" -X kill

# stopping ray
conda run -n adv-tf2 ray stop
