#!/usr/bin/env bash
# ===========================================================================
# Train one Isaac Lab task on the Newton backend, fixed vs adaptive.
# Self-contained experiment folder: this script + env.py (the env) + scenes/ live together.
#
# New experiment:  cp -r _template <name>   then edit <name>/env.py and the EDIT block below.
#   SOLVER=fixed    VIDEO=1 bash <name>/train.sh
#   SOLVER=adaptive VIDEO=1 bash <name>/train.sh
# Fixed + adaptive land in the same W&B project/group, tagged by solver -> overlay
# Train/mean_reward in the UI.
# ===========================================================================
set -euo pipefail

# ===================== EDIT FOR THIS EXAMPLE =====================
TASK="${TASK:-Isaac-Example-Rubato}"   # the gym id registered in this folder's env.py
GROUP="${GROUP:-example}"               # W&B group: pairs THIS example's fixed+adaptive runs
# ================================================================

ISAACLAB="${ISAACLAB:-$HOME/Documents/code/IsaacLab}"
LAB_SH="$ISAACLAB/isaaclab.sh"   # supported entry: resolves the venv + dispatches the train CLI
[ -x "$LAB_SH" ] || { echo "isaaclab.sh not found at $LAB_SH (set \$ISAACLAB)" >&2; exit 1; }

# This folder holds env.py -> put it on PYTHONPATH and load it via train.py's --external_callback.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
REGISTER="${REGISTER:-env.register}"   # env.py:register(); set empty to run a pure stock task id

SOLVER="${SOLVER:-adaptive}"                 # fixed | adaptive  <- the one-flag difference
SEED="${SEED:-42}"
# W&B SETUP (once per machine): `wandb login` (or export WANDB_API_KEY=...) + export WANDB_ENTITY=<you>.
# Never commit the key -- .gitignore covers wandb/.
WANDB_PROJECT="${WANDB_PROJECT:-newton-adaptive-study}"
VIDEO="${VIDEO:-0}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
NUM_ENVS="${NUM_ENVS:-}"            # unset -> task cfg default (keeps both solvers matched)
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
[ -n "$REGISTER" ]       && OPT_ARGS+=(--external_callback "$REGISTER")
[ -n "$NUM_ENVS" ]       && OPT_ARGS+=(--num_envs "$NUM_ENVS")
[ -n "$MAX_ITERATIONS" ] && OPT_ARGS+=(--max_iterations "$MAX_ITERATIONS")
[ "$VIDEO" = "1" ]       && OPT_ARGS+=(--video --video_interval "$VIDEO_INTERVAL" --video_length "$VIDEO_LENGTH")

echo "==> $TASK | solver=$TAG | seed=$SEED | wandb=$WANDB_PROJECT/$RUN_NAME group=$WANDB_RUN_GROUP (${WANDB_MODE:-online})"
[ "$TAG" = adaptive ] && echo "==> adaptive telemetry -> ${NEWTON_ADAPTIVE_LOG:-/tmp/newton_adaptive.log}"

# isaaclab.sh train (NOT the deprecated scripts/.../rsl_rl/train.py); args pass straight through
exec env "${ADAPTIVE_ENV[@]}" "$LAB_SH" train --rl_library rsl_rl \
  --task "$TASK" physics=newton_mjwarp \
  --headless --seed "$SEED" \
  --logger wandb --log_project_name "$WANDB_PROJECT" --run_name "$RUN_NAME" \
  "${OPT_ARGS[@]}"
