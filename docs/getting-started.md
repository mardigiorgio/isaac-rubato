# Getting started

## Requirements

- Ubuntu, NVIDIA GPU with driver ≥ 580 (the RTX 5090 needs a Blackwell-capable driver), ~60 GB free.
- [`uv`](https://docs.astral.sh/uv/) and `git` on PATH.

## Install

```bash
git clone --recurse-submodules https://github.com/mardigiorgio/isaac-rubato.git
cd isaac-rubato
bash install/setup.sh
```

`setup.sh` reconstructs the full environment from pinned upstreams — it does **not** vendor the 30 GB
Isaac Sim wheel or Isaac Lab:

1. clones Isaac Lab at a pinned commit,
2. creates a Python 3.12 venv and installs the Isaac Sim 6.0 wheel + the pinned torch (cu128),
3. installs Isaac Lab editable and overrides its Newton with the `newton-adaptive` submodule,
4. applies the adaptive delta (the `isaaclab_newton` patch + the GUI-toggle extension),
5. verifies that `SolverMuJoCoAdaptive` is live.

## Open the interactive editor

```bash
rubato
```

Launches the Isaac Sim editor on the Newton backend (the `isaacsim.exp.full.newton` experience). Drop an
object → **Play → it falls on Newton.** The window title reads *Isaac Sim Full (Newton Physics)*.

## Run a task with the adaptive solver

```bash
NEWTON_ADAPTIVE=1 NEWTON_ADAPTIVE_LOG_EVERY=10 \
  demos/trossen/run_native.sh demos/trossen/train_teacher.py --headless --num_envs 16
```

Then watch the solver subdivide `dt` in real time:

```bash
tail -f /tmp/newton_adaptive.log
```

`spread > 0` and substeps-per-frame well above ~3 means the adaptive solver is doing real work under
contact. Drop `NEWTON_ADAPTIVE=1` to run stock Newton instead.

## Selecting the adaptive solver

Three equivalent switches, all read when the solver is built:

=== "Config"

    ```python
    MJWarpSolverCfg(adaptive=True, adaptive_dt_init=0.005, adaptive_dt_mode="per_world")
    ```

=== "Environment variable"

    ```bash
    NEWTON_ADAPTIVE=1   # + NEWTON_ADAPTIVE_TOL, NEWTON_ADAPTIVE_DTMODE, NEWTON_ADAPTIVE_DT_INIT
    ```

=== "GUI toggle"

    The **Adaptive timestepping** checkbox (the `newton_adaptive_ui` extension) sets the
    `/isaaclab/newton/adaptive` carb setting; the solver rebuilds adaptive on the next Stop → Play.

## View this documentation locally

```bash
uvx --with mkdocs-material mkdocs serve
```

Opens an interactive, searchable site at <http://127.0.0.1:8000> with live reload.
