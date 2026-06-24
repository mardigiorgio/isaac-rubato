# isaac-rubato

isaac-rubato is a modified [Isaac Sim](https://developer.nvidia.com/isaac/sim) /
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) / [Newton](https://github.com/newton-physics/newton)
stack: the standard Isaac toolchain with an adaptive-timestepping solver wired into Isaac Lab's Newton
backend as a selectable option, so reinforcement-learning policies can be trained on adaptive-step
(rather than fixed-step) physics. It is the Isaac stack plus that integration, not a standalone platform.

*Rubato* (It., "stolen") is the flexible variation of local tempo in music. Here it names adaptive
timestepping, where integration effort is reallocated across a frame (spent at stiff contact, withheld in
free motion) while the frame, and with it the fixed-rate control boundary, is preserved.

## How it works

Isaac Lab steps physics through a single manager class (`NewtonMJWarpManager`, in its `isaaclab_newton`
extension) that builds and steps the Newton solver each control interval. The integration is two methods
on that class (solver construction and solver stepping), so the adaptive solver (`SolverMuJoCoAdaptive`,
error-controlled step-doubling over MuJoCo-Warp) is selectable alongside PhysX and stock Newton, with the
policy, rewards, observations, and rendering unchanged. The control interface is preserved exactly: the
action is zero-order-held across the interval, and the solver subdivides time only *within* the interval,
always landing on the boundary.

## Installation

Prerequisites you install yourself: [`uv`](https://docs.astral.sh/uv/getting-started/installation/), an NVIDIA driver >= 580, `git`, and ~60 GB free disk.

```bash
# clone the platform, the custom Newton fork (beside it), and Isaac Lab at the pinned commit
git clone https://github.com/mardigiorgio/isaac-rubato.git && cd isaac-rubato
git clone https://github.com/mardigiorgio/newton-adaptive.git ../newton-adaptive
git clone https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
git -C ../IsaacLab checkout 546551f5ba8e8e4fbbcbf589b63c6f40b7cacb3f

# build the venv and install the locked platform (Isaac Sim + PyTorch cu128 + the Newton fork)
uv venv --python 3.12 .venv
uv sync --locked

# install Isaac Lab into the same venv, then re-assert the fork over Isaac Lab's stock Newton
VIRTUAL_ENV="$PWD/.venv" OMNI_KIT_ACCEPT_EULA=YES ../IsaacLab/isaaclab.sh -i
uv sync --inexact --locked

# apply the adaptive delta to Isaac Lab, then verify
ISAACLAB="$PWD/../IsaacLab" bash integration/apply_isaaclab_delta.sh ../IsaacLab
uv run python install/verify.py
```

`verify.py` confirms the Newton fork is the active import and `SolverMuJoCoAdaptive` is present. Update the fork later with `git -C ../newton-adaptive pull && uv lock --upgrade-package newton && uv sync --inexact --locked`, then re-run the delta step.

## Getting started

Open the Isaac Sim editor on the Newton backend (from the repo root):

```bash
./isaac-rubato
```

In any Isaac Lab task, the adaptive solver is selected three ways: the config flag
`MJWarpSolverCfg(adaptive=True)`, the env var `NEWTON_ADAPTIVE=1`, or the GUI toggle (the
`newton_adaptive_ui` Kit extension). Setting `NEWTON_ADAPTIVE_LOG_EVERY=N` writes the per-frame dt and
sub-step counts to `/tmp/newton_adaptive.log`. Drop `NEWTON_ADAPTIVE` to run stock Newton.

A minimal cartpole training demo will live under `demos/` once the adaptive backend is dialed in.

## Built on

Layered on three upstream projects, used under their own licenses:

- **Isaac Sim**: NVIDIA Omniverse robotics simulator. <https://developer.nvidia.com/isaac/sim>
- **Isaac Lab**: NVIDIA RL / robot-learning framework (BSD-3-Clause). <https://github.com/isaac-sim/IsaacLab>
- **Newton**: GPU physics over MuJoCo-Warp (Apache-2.0). <https://github.com/newton-physics/newton>

This repository contributes the adaptive-solver integration, the RL workstream, the scenes, and the
install automation.
