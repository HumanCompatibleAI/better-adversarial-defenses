# better-adversarial-defenses
Trying to improve on the original publication, intern project 2020 by Sergei Volodin

Running:
1. Install and run mongodb with database `chai`
2. Install miniconda anc create two environments: tf1 and tf2

To make video:
python make_video.py --checkpoint path/to/checkpoint/checkpoint-xxx --display $DISPLAY
Options: `--load_normal True` to evaluate against normal opponent instead of the trained one
`--steps n` number of steps to make (1 is `256`steps which is approximately 1 episode)
`--no_video True` will disable video. Use this to evaluate the performance with more episodes

To train:
`python train.py --tune xxx` where `xxx` is one of the settings in `config.py` from `CONFIGS` variable.

Running Stable Baselines server:
`(tf1) $ python frankenstein/stable_baselines_server.py`

Training in Inverted Pendulum (single-agent):
`python train.py --tune external_cartpole`
`python make_video.py --checkpoint /home/sergei/ray_results/External_multicomp_2020-09-09_21-15-40f269e03n/checkpoint_13/checkpoint-13 --display $DISPLAY --config external_cartpole --steps 1`

Log files will appear in `~/ray_results/run_type/run_name`. Use tensorboard in this folder. Checkpoints will be in `~/ray_results/iteration_name` where `iteration_name` can be obtained from `run_type/run_name` in variable `checkpoint_rllib`.

## Rendering in MuJoCo
1. Install MuJoCo 1.13
2. Run Xvfb: `Xvfb -screen 0 1024x768x24`
3. `export DISPLAY=:0`, `env.render(mode='rgb_array')` will give the image

Install gym_compete and aprl via setup.py
Install ray 0.8.6 and then setup ray in development mode

Installing fonts:
`mkdir ~/.fonts; cp $CONDA_PREFIX/fonts/*.ttf ~/.fonts; fc-cache -f -v`
