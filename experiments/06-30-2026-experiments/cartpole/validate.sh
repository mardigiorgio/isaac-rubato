#!/usr/bin/env bash
# Keyless validation of the ADAPTIVE Newton path on cartpole: short, headless, 64 envs,
# 50 iters, W&B forced OFFLINE so it runs WITHOUT an API key (offline runs land in ./wandb/).
# PASS = exit 0 with a populated /tmp/newton_adaptive.log (per-frame inner-dt / substep
# telemetry) -> the adaptive solver built, stepped, and trained cleanly on cartpole.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOLVER=adaptive NUM_ENVS=64 MAX_ITERATIONS=50 SEED=42 \
RUN_NAME=cartpole-adaptive-validate WANDB_PROJECT=newton-adaptive-study \
WANDB_MODE=offline NEWTON_ADAPTIVE_LOG_EVERY=10 \
  bash "$HERE/train.sh"

echo
echo "==================== adaptive telemetry (tail) ===================="
if [ -s /tmp/newton_adaptive.log ]; then
  tail -n 20 /tmp/newton_adaptive.log
  echo "=================================================================="
  echo "OK: telemetry populated -> the adaptive solver built and stepped on cartpole."
else
  echo "WARNING: /tmp/newton_adaptive.log empty/absent -> NEWTON_ADAPTIVE or physics=newton_mjwarp"
  echo "  was not picked up; the adaptive solver was never built. Check the run output above." >&2
  exit 1
fi
