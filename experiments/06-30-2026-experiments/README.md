# 06-30-2026 experiments

Fixed vs. adaptive Newton on a set of Isaac Lab tasks. Cartpole (smoke), franka-reach, and the
contact-rich ones to come (cube-reorient, lift, ...).

## Layout - one self-contained folder per experiment

```
_template/          copy this whole folder to start a new experiment
  env.py            the env: subclass a stock task, add edits, register a gym id
  train.sh          fixed-vs-adaptive runner
cartpole/           experiment #1 (smoke)
  env.py  train.sh  validate.sh  README.md
franka-reach/
  env.py  train.sh  README.md
  scenes/           GUI-built .usd scenes for THIS experiment
```

Each experiment is everything-in-one-folder: its `env.py` (the env, with your edits), its
`train.sh`, its scenes, its notes. No shared package, no loose scripts - copy a folder, you have a
new experiment; `git pull`, the whole thing comes with you.

**How the env loads:** `train.sh` puts its own folder on `PYTHONPATH` and passes
`--external_callback env.register` to `isaaclab.sh train`, which registers the folder's gym id
before the task is built. Nothing is installed - `env.py` is imported straight from the repo.

## New experiment

```bash
cp -r _template franka-stack
# edit franka-stack/env.py  (import + subclass the stock task, register the id)
# edit the EDIT block in franka-stack/train.sh  (TASK + GROUP)
SOLVER=fixed    VIDEO=1 bash franka-stack/train.sh
SOLVER=adaptive VIDEO=1 bash franka-stack/train.sh
```

## Day to day

Edit a folder's `env.py`, or save a GUI scene into its `scenes/`. Commit. `git pull` on the other
machine - it's there. **No reinstall**: envs are read from the repo at runtime. The only
per-machine setup is the platform itself (Isaac Lab + adaptive Newton + the Newton fork).

Before running, activate the platform venv once per shell so `isaaclab.sh` uses it:
`source <repo>/.venv/bin/activate`.

## W&B setup (once per machine)

```bash
wandb login                      # paste your key  (or: export WANDB_API_KEY=...)
export WANDB_ENTITY=<your-wandb-entity>
```
Runs log to `newton-adaptive-study` (override `WANDB_PROJECT`). Keyless? prefix `WANDB_MODE=offline`
(that's what `cartpole/validate.sh` does). Never commit the key - `.gitignore` covers `wandb/`.

Each `train.sh` calls `isaaclab.sh train --rl_library rsl_rl`. Fixed + adaptive share a W&B group,
tagged by solver - overlay `Train/mean_reward` in the UI.

## Known gaps

- **Goal marker in reach videos.** Stock goal marker is debug-vis and doesn't render headless. A
  real blue sphere needs a pure-visual scene prim moved by a transform view (a `RigidObject` is
  rejected by the Newton body parser). Deferred.
