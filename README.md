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

isaac-rubato is the Isaac stack (Isaac Sim 6.0 + Isaac Lab) with the adaptive Newton solver wired in. The heavy pip layer (Isaac Sim, PyTorch cu128, the Newton fork) is consolidated in a `uv` lockfile, so it installs with one command. Isaac Lab is a separate clone installed by its own script into the same venv, then the adaptive delta is applied on top.

Plan for ~30-60 min and a fast connection: the Isaac Sim wheel set alone is several GB.

### Prerequisites

Install these yourself first:

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — the Python package/venv manager that drives the install.
- **NVIDIA driver >= 580** — Isaac Sim 6.0 and the cu128 PyTorch build require it (Blackwell GPUs need a recent driver). Check with `nvidia-smi`.
- [`git`](https://git-scm.com/downloads).
- **~60 GB free disk** — the Isaac Sim wheel + extscache and the editable checkouts are large.

### Steps

**1. Clone isaac-rubato.**

```bash
git clone https://github.com/mardigiorgio/isaac-rubato.git
cd isaac-rubato
```

This repo holds `pyproject.toml` + `uv.lock`, the adaptive delta, and the `rubato` launcher. Isaac Sim and Isaac Lab are not vendored — they are reconstructed from pinned upstreams below.

**2. Clone the custom Newton fork *next to* this repo.**

```bash
git clone https://github.com/mardigiorgio/newton-adaptive.git ../newton-adaptive
```

This is the adaptive-solver fork (`SolverMuJoCoAdaptive`, error-controlled step-doubling over MuJoCo-Warp). `pyproject.toml` declares it as a **path-editable** source at `../newton-adaptive`, so the checkout must exist here *before* `uv lock`/`uv sync` runs (this is why it is cloned before the sync, not after). `uv sync` then installs it editable from this checkout — the working tree is the live import path. → [newton-adaptive](https://github.com/mardigiorgio/newton-adaptive)

**3. Clone Isaac Lab at the pinned commit.**

```bash
git clone https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
git -C ../IsaacLab checkout 546551f5ba8e8e4fbbcbf589b63c6f40b7cacb3f
```

This is the Isaac Lab `develop` commit the adaptive delta was generated against. Pinning it keeps the patch applying cleanly. → [Isaac Lab](https://github.com/isaac-sim/IsaacLab)

**4. Create the venv and sync the platform from the lock.**

```bash
uv venv --python 3.12 .venv
uv sync --locked
```

The one command that does the heavy lifting: `uv sync` reads `uv.lock` and installs Isaac Sim 6.0.0.1 (NVIDIA index), `torch`/`torchvision` +cu128 (PyTorch index), and the editable Newton fork from `../newton-adaptive` — exact pinned versions and hashes, no flags to remember. `--locked` fails fast if the lock is stale instead of silently re-resolving. → [uv sync](https://docs.astral.sh/uv/concepts/projects/sync/)

**5. Install Isaac Lab into the same venv.**

```bash
VIRTUAL_ENV="$PWD/.venv" OMNI_KIT_ACCEPT_EULA=YES ../IsaacLab/isaaclab.sh -i
```

Isaac Lab is a monorepo of 15 editable sub-packages with its own installer (symlinks, RL-framework extras, rsl-rl wiring) that the lock cannot drive — so it runs as its own step. `isaaclab.sh` auto-detects `$VIRTUAL_ENV/bin/python` and installs everything editable into the venv `uv` just built. It also pulls a *stock* upstream Newton, which can shadow the fork — so immediately re-assert the override:

```bash
uv sync --locked   # re-install the editable path fork on top, last writer wins
```

After this, `import newton` loads the fork (same dist name `newton`, but the editable path install is the last to touch site-packages). → [Isaac Lab install docs](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)

**6. Apply the adaptive delta to Isaac Lab.**

```bash
ISAACLAB="$PWD/../IsaacLab" bash integration/apply_isaaclab_delta.sh ../IsaacLab
```

Patches Isaac Lab's Newton backend to expose `SolverMuJoCoAdaptive` as a selectable solver and drops in the `newton_adaptive_ui` GUI-toggle extension. Idempotent — safe to re-run. → [integration/INTEGRATION.md](integration/INTEGRATION.md)

**7. Verify.**

```bash
uv run python install/verify.py
```

Asserts the Newton fork is the active import (not Isaac Lab's stock pin), that `SolverMuJoCoAdaptive` is present, and that `isaaclab` imports. If all three pass, the adaptive backend is wired in. If verify reports the stock Newton is active, re-run `uv sync --locked` (step 5's re-sync) to restore the override.

### Updating the platform

```bash
git -C ../newton-adaptive pull        # update the fork working tree (live import path)
uv lock --upgrade-package newton      # re-resolve the path-editable fork
uv sync --locked                      # re-install to match
```

Because the fork is a path-editable source, a plain `git pull` in `../newton-adaptive` already updates the live import path; `uv lock --upgrade-package newton` re-resolves its metadata and rewrites `uv.lock`. After updating the fork, re-run step 6 to re-apply the delta.

## Getting started

Open the Isaac Sim editor on the Newton backend:

```bash
rubato
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
