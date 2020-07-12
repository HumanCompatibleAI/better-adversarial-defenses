# better-adversarial-defenses
Trying to improve on the original publication, intern project 2020 by Sergei Volodin

## Rendering in MuJoCo
1. Install MuJoCo 1.13
2. Run Xvfb: `Xvfb -screen 0 1024x768x24`
3. `export DISPLAY=:0`, `env.render(mode='rgb_array')` will give the image