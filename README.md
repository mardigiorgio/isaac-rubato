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
# clone the platform, the custom Newton fork, the SAP solver, and Isaac Lab at the pinned commit
git clone https://github.com/mardigiorgio/isaac-rubato.git && cd isaac-rubato
git clone https://github.com/mardigiorgio/newton-adaptive.git ../newton-adaptive
git clone https://github.com/mardigiorgio/sap_warp.git ../sap_warp
git clone https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
git -C ../IsaacLab checkout 546551f5ba8e8e4fbbcbf589b63c6f40b7cacb3f

# sap_warp (the SAP / SAP-adaptive convex-contact solver) has no installable package: the Newton fork
# adds its root to sys.path at import time. SAP_WARP_PATH points at that clone; export it (in the shell
# that runs training/editor) when sap_warp is not at the default sibling path below.
export SAP_WARP_PATH="$PWD/../sap_warp"

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

Activate the platform venv first, so `isaaclab.sh` and the launcher use it (without this, a fresh
install falls back to the wrong Python):

```bash
source .venv/bin/activate     # Isaac Sim + Isaac Lab + the Newton fork
```

**Editor** - build/inspect scenes on the Newton backend (from the repo root):

```bash
./isaac-rubato
```

**Training** - the fixed-vs-adaptive experiments live in `experiments/`. Each is a self-contained
folder (its `env.py` + run script); envs are loaded from the repo at runtime, nothing extra to install:

```bash
cd experiments/06-30-2026-experiments
bash cartpole/validate.sh                       # keyless smoke -- proves the adaptive path trains
SOLVER=fixed    VIDEO=1 bash franka-reach/train.sh
SOLVER=adaptive VIDEO=1 bash franka-reach/train.sh
```

### Selecting the solver (`--solver`)

The Newton backend exposes four solver variants through the `--solver` flag on the stock `isaaclab.sh
train` entry point (added by `integration/cli_solver_flag.patch`, applied by the delta script). It drives
the `MJWarpSolverCfg` latches (`backend` / `adaptive` / `sap_adaptive`) read by `NewtonMJWarpManager`:

| `--solver` | backend | constructs | notes |
|---|---|---|---|
| `mujoco` | MuJoCo-Warp | `SolverMuJoCo` | fixed-step (stock Newton default) |
| `mujoco-adaptive` | MuJoCo-Warp | `SolverMuJoCoAdaptive` | error-controlled step-doubling |
| `sap` | SAP (`sap_warp`) | `SolverSAP` | fixed-step convex compliant contact |
| `sap-adaptive` | SAP (`sap_warp`) | `SolverSAPAdaptive` | step-doubling SAP (even + global tiling) |

The two `sap*` variants require the `sap_warp` clone on `SAP_WARP_PATH` (see install). Run the cube-reorient
study task (env on `PYTHONPATH`, registered via `--external_callback env.register`) with, e.g.:

```bash
export PYTHONPATH="$PWD/shadow-hand-repose-cube"
./../../../IsaacLab/isaaclab.sh train --solver sap-adaptive \
  --task Isaac-Reorient-Cube-Shadow-Rubato --external_callback env.register \
  --rl_library rsl_rl --headless --num_envs 64 --max_iterations 1 physics=newton_mjwarp
# swap --solver for mujoco | mujoco-adaptive | sap | sap-adaptive
```

The experiment `train.sh` scripts still select MuJoCo fixed-vs-adaptive via `SOLVER=fixed|adaptive`
(the `NEWTON_ADAPTIVE=1` env var). The adaptive paths (`mujoco-adaptive`, `sap-adaptive`) also respond to
the config flag `MJWarpSolverCfg(adaptive=True)` and the GUI toggle (`newton_adaptive_ui` Kit extension);
`NEWTON_ADAPTIVE_LOG_EVERY=N` writes per-frame dt + sub-step counts to `/tmp/newton_adaptive.log`.

## Built on

Layered on three upstream projects, used under their own licenses:

- **Isaac Sim**: NVIDIA Omniverse robotics simulator. <https://developer.nvidia.com/isaac/sim>
- **Isaac Lab**: NVIDIA RL / robot-learning framework (BSD-3-Clause). <https://github.com/isaac-sim/IsaacLab>
- **Newton**: GPU physics over MuJoCo-Warp (Apache-2.0). <https://github.com/newton-physics/newton>

This repository contributes the adaptive-solver integration, the RL workstream, the scenes, and the
install automation.
