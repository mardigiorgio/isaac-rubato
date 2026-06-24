# Cartpole - Newton adaptive vs fixed (experiment #1)

Example #1 of the adaptive-solver study, and the validation smoke. Cartpole is a trivial control
task (no stiff contact), so it exists to prove the adaptive Newton path trains a competent policy
cleanly before we spend compute on hard tasks. The result to read is **reward parity**: fixed-step
and adaptive-step Newton should learn the same curve.

Honest note: even on cartpole the adaptive controller still subdivides (~12-37 substeps/frame,
inner-dt spread ~4-8e-3 s), because the default `dt_inner_init=0.01` is larger than the per-frame
substep. That is config-driven subdivision, not stiff-contact adaptation - so adaptive costs more
compute here too. Example #1 measures whether reward parity holds despite that overhead.

These scripts are thin wrappers around Isaac Lab's stock `rsl_rl` trainer. Set `$ISAACLAB` if your
Isaac Lab clone is not at `~/Documents/code/IsaacLab`.

## W&B setup (once)

```bash
$ISAACLAB/env_isaaclab/bin/wandb login     # or: export WANDB_API_KEY=<key from wandb.ai/authorize>
export WANDB_ENTITY=<your-wandb-entity>     # your wandb user or team
```

Runs land in project `newton-adaptive-study`.

## Run

```bash
SOLVER=adaptive VIDEO=1 bash train.sh        # adaptive (+ records a training timelapse)
SOLVER=fixed    VIDEO=1 bash train.sh        # fixed baseline (same task/seed/iters)
```

Both log to W&B automatically: `Train/mean_reward` (overlay fixed vs adaptive for the parity
result), `Perf/total_fps` (the adaptivity overhead), and the `--video` timelapses (W&B "video"
media panel). Adaptive also writes dt/substep telemetry to `/tmp/newton_adaptive.log`. The
`WANDB_RUN_GROUP=cartpole` tag pairs the two runs in the UI.

## Validate (no W&B key needed)

```bash
bash validate.sh    # short adaptive run, W&B offline, confirms it trains clean
```

## The 4 examples

Each example is a matched fixed-vs-adaptive pair - identical task, seed, and PPO hyperparameters,
differing only in the solver - so the curves overlay directly in W&B.

| # | task | why |
|---|------|-----|
| 1 | cartpole (this) | trivial control - clean-train smoke + reward-parity baseline |
| 2-4 | contact-rich tasks (TBD) | where adaptive should actually pay off (stiff contact) |
