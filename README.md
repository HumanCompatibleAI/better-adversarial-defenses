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

## Rendering in MuJoCo
1. Install MuJoCo 1.13
2. Run Xvfb: `Xvfb -screen 0 1024x768x24`
3. `export DISPLAY=:0`, `env.render(mode='rgb_array')` will give the image

Installing fonts:
`mkdir ~/.fonts; cp $CONDA_PREFIX/fonts/*.ttf ~/.fonts; fc-cache -f -v`
