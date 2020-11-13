#!/bin/bash

# exit when any command fails
set -e

# possibly starting Xvfb
if [ "X$DISPLAY" == "X" ]
then
	echo "Launching Xvfb..."
	screen -Sdm "xvfb_test" Xvfb -screen 0 1024x768x24
	export DISPLAY=:0
fi

if [ "X$MJKEY_NEW" != "X" ]
then
    echo "$MJKEY_NEW" > ~/.mujoco/mjkey.txt
fi


if [ "X$1" != "X--no_mujoco_license" ]
then
    bash ./mjkey-prompt.sh
fi

# Disabling GPU
export CUDA_VISIBLE_DEVICES=-1

# launching mongodb
if  [ "$(netstat -tulpn|grep 27017|wc -l)" == "0" ]
then
    echo "Starting mongo"
    sudo service mongodb start
fi

# running pytest

# runs without mujoco
echo "Testing weights load from pickle files (tf1)"
conda run -n adv-tf1 pytest -s -v gym_compete_rllib/test_load_gym_compete_policy.py::test_load

# runs without mujoco
echo "Testing weights load from pickle files (tf2)"
conda run -n adv-tf2 pytest -s -v gym_compete_rllib/test_load_gym_compete_policy.py::test_load

# runs without mujoco
echo "Testing ap_rllib"
conda run -n adv-tf2 pytest -s -v ap_rllib

if [ "X$1" != "X--no_mujoco_license" ]
then
	# does not run without mujoco
	echo "Testing zoo networks consistency"
	conda run -n adv-tf1 pytest -s -v gym_compete_rllib/test_load_gym_compete_policy.py::test_prediction


	echo "Testing environment"
	conda run -n adv-tf2 pytest -s -v gym_compete_rllib/test_load_rllib_env.py
fi

# launching the stable baselines server
echo "Launching SB server..."
screen -Sdm "sb_server_test" conda run -n adv-tf1 python -m frankenstein.stable_baselines_server

if [ "X$1" != "X--no_mujoco_license" ]
then
	# launching training with a few iterations
	echo "Running training..."
	conda run -n adv-tf2 python -m ap_rllib.train --tune external_test

	# obtaining the checkpoint
	checkpoint=$(conda run -n adv-tf2 python -m ap_rllib_experiment_analysis.get_last_checkpoint --config external_test)

	# making a video
	echo "Making videos..."
	conda run -n adv-tf2 python -m ap_rllib.make_video --config external_test --checkpoint $checkpoint --display $DISPLAY --steps 1
fi

sleep 5

# closing the screen session
echo "Stopping sb server"
screen -S "sb_server_test" -X kill || true
screen -S "xvfb_test" -X kill || true

# stopping ray
echo "Stopping ray and Xvfb"
conda run -n adv-tf2 ray stop || true
pkill -f -9 ray || true

# final message
echo "All tests PASSED!"
