#!/usr/bin/env bash
# ===========================================================================
# TEMPLATE: train one Isaac Lab task on the Newton backend, fixed vs adaptive.
#
# To make a new example:
#   mkdir 06-30-2026-experiments/<name>
#   cp 06-30-2026-experiments/_template.sh 06-30-2026-experiments/<name>/train.sh
#   # edit the two lines under "EDIT FOR THIS EXAMPLE", then:
#   SOLVER=fixed    VIDEO=1 bash 06-30-2026-experiments/<name>/train.sh
#   SOLVER=adaptive VIDEO=1 bash 06-30-2026-experiments/<name>/train.sh
#
# Fixed and adaptive land in the SAME W&B project, grouped together, tagged by
# solver -- overlay Train/mean_reward in the UI. (validate.sh from cartpole is
# task-agnostic: it just calls ./train.sh, so copy it as-is for a keyless smoke.)
# ===========================================================================
set -euo pipefail

# ===================== EDIT FOR THIS EXAMPLE =====================
TASK="${TASK:-Isaac-Cartpole}"   # the Isaac Lab gym task id (grep gym.register in source/isaaclab_tasks)
GROUP="${GROUP:-cartpole}"        # W&B group: pairs THIS example's fixed+adaptive runs
# ================================================================

ISAACLAB="${ISAACLAB:-$HOME/Documents/code/IsaacLab}"
PYTHON="$ISAACLAB/env_isaaclab/bin/python"
TRAIN_PY="$ISAACLAB/scripts/reinforcement_learning/rsl_rl/train.py"
[ -x "$PYTHON" ]   || { echo "venv python not found at $PYTHON (set \$ISAACLAB)" >&2; exit 1; }
[ -f "$TRAIN_PY" ] || { echo "train.py not found at $TRAIN_PY" >&2; exit 1; }

SOLVER="${SOLVER:-adaptive}"                 # fixed | adaptive  <- the one-flag difference
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-newton-adaptive-study}"
VIDEO="${VIDEO:-0}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
# Leave NUM_ENVS / MAX_ITERATIONS unset to use the task's own cfg defaults
# (both solvers get the same defaults, so the comparison stays matched).
NUM_ENVS="${NUM_ENVS:-}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"

case "$SOLVER" in
  fixed)    ADAPTIVE_ENV=(); TAG=fixed ;;
  adaptive) rm -f "${NEWTON_ADAPTIVE_LOG:-/tmp/newton_adaptive.log}"
            ADAPTIVE_ENV=("NEWTON_ADAPTIVE=1" "NEWTON_ADAPTIVE_LOG_EVERY=${NEWTON_ADAPTIVE_LOG_EVERY:-10}"); TAG=adaptive ;;
  *) echo "SOLVER must be fixed|adaptive (got '$SOLVER')" >&2; exit 1 ;;
esac

RUN_NAME="${RUN_NAME:-$GROUP-$TAG-s$SEED}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$GROUP}"   # pairs fixed+adaptive in the W&B UI
export WANDB_TAGS="${WANDB_TAGS:-$TAG}"

OPT_ARGS=()
[ -n "$NUM_ENVS" ]       && OPT_ARGS+=(--num_envs "$NUM_ENVS")
[ -n "$MAX_ITERATIONS" ] && OPT_ARGS+=(--max_iterations "$MAX_ITERATIONS")
[ "$VIDEO" = "1" ]       && OPT_ARGS+=(--video --video_interval "$VIDEO_INTERVAL" --video_length "$VIDEO_LENGTH")

echo "==> $TASK | solver=$TAG | seed=$SEED | wandb=$WANDB_PROJECT/$RUN_NAME group=$WANDB_RUN_GROUP (${WANDB_MODE:-online})"
[ "$TAG" = adaptive ] && echo "==> adaptive telemetry -> ${NEWTON_ADAPTIVE_LOG:-/tmp/newton_adaptive.log}"

cd "$ISAACLAB"
exec env "${ADAPTIVE_ENV[@]}" "$PYTHON" "$TRAIN_PY" \
  --task "$TASK" physics=newton_mjwarp \
  --headless --seed "$SEED" \
  --logger wandb --log_project_name "$WANDB_PROJECT" --run_name "$RUN_NAME" \
  "${OPT_ARGS[@]}"
