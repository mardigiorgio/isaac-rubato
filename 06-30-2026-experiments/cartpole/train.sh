#!/usr/bin/env bash
# Cartpole on the Newton backend: fixed-step vs adaptive-step solver.
# Thin wrapper around Isaac Lab's stock rsl_rl trainer -- no training logic here.
# fixed vs adaptive is one flag:   SOLVER=fixed | SOLVER=adaptive
# Set $ISAACLAB if your Isaac Lab clone is not at ~/Documents/code/IsaacLab.
set -euo pipefail

ISAACLAB="${ISAACLAB:-$HOME/Documents/code/IsaacLab}"
PYTHON="$ISAACLAB/env_isaaclab/bin/python"
TRAIN_PY="$ISAACLAB/scripts/reinforcement_learning/rsl_rl/train.py"
[ -x "$PYTHON" ]   || { echo "venv python not found at $PYTHON (set \$ISAACLAB)" >&2; exit 1; }
[ -f "$TRAIN_PY" ] || { echo "train.py not found at $TRAIN_PY" >&2; exit 1; }

SOLVER="${SOLVER:-adaptive}"             # fixed | adaptive  <- the one-flag difference
SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-150}"  # cartpole default
WANDB_PROJECT="${WANDB_PROJECT:-newton-adaptive-study}"
VIDEO="${VIDEO:-0}"                      # 1 = record + auto-upload timelapse clips to W&B
VIDEO_INTERVAL="${VIDEO_INTERVAL:-200}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"

case "$SOLVER" in
  fixed)    ADAPTIVE_ENV=(); TAG=fixed ;;
  adaptive) rm -f "${NEWTON_ADAPTIVE_LOG:-/tmp/newton_adaptive.log}"
            ADAPTIVE_ENV=("NEWTON_ADAPTIVE=1" "NEWTON_ADAPTIVE_LOG_EVERY=${NEWTON_ADAPTIVE_LOG_EVERY:-10}"); TAG=adaptive ;;
  *) echo "SOLVER must be fixed|adaptive (got '$SOLVER')" >&2; exit 1 ;;
esac

RUN_NAME="${RUN_NAME:-cartpole-$TAG-s$SEED}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-cartpole}"   # pairs fixed+adaptive in the W&B UI
export WANDB_TAGS="${WANDB_TAGS:-$TAG}"

VIDEO_ARGS=()
[ "$VIDEO" = "1" ] && VIDEO_ARGS=(--video --video_interval "$VIDEO_INTERVAL" --video_length "$VIDEO_LENGTH")

echo "==> cartpole | solver=$TAG | seed=$SEED | iters=$MAX_ITERATIONS | envs=$NUM_ENVS | wandb=$WANDB_PROJECT/$RUN_NAME (${WANDB_MODE:-online})"
[ "$TAG" = adaptive ] && echo "==> adaptive telemetry -> ${NEWTON_ADAPTIVE_LOG:-/tmp/newton_adaptive.log}"

cd "$ISAACLAB"
exec env "${ADAPTIVE_ENV[@]}" "$PYTHON" "$TRAIN_PY" \
  --task Isaac-Cartpole physics=newton_mjwarp \
  --headless --num_envs "$NUM_ENVS" --seed "$SEED" --max_iterations "$MAX_ITERATIONS" \
  --logger wandb --log_project_name "$WANDB_PROJECT" --run_name "$RUN_NAME" \
  "${VIDEO_ARGS[@]}"
