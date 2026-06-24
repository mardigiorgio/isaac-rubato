#!/usr/bin/env bash
# Keyless validation of the ADAPTIVE Newton path on cartpole.
# Short, headless, small (64 envs, 50 iters). W&B forced OFFLINE so it runs WITHOUT
# an API key (offline runs land in ./wandb/, sync later with `wandb sync`).
# Adaptive telemetry -> /tmp/newton_adaptive.log.
#
# PASS = exit 0 with finite rewards + a populated /tmp/newton_adaptive.log (per-frame
# inner-dt / substep telemetry). That proves the adaptive solver builds, steps, and trains
# cleanly on cartpole. (The "SolverMuJoCoAdaptive built" confirmation is in the run's stdout,
# via logger.info -- not in the telemetry file.)
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
  echo "NOTE: even on cartpole the default controller subdivides (dt_inner_init=0.01 is"
  echo "      larger than the frame substep), so expect inner_dt spread ~4-8e-3 s and"
  echo "      ~12-37 substeps/frame. That is config-driven subdivision, NOT stiff-contact"
  echo "      adaptation. The smoke's point is a clean train; compare reward vs a fixed run."
else
  echo "WARNING: /tmp/newton_adaptive.log empty/absent -> NEWTON_ADAPTIVE or physics=newton_mjwarp"
  echo "  was not picked up; the adaptive solver was never built. Check the run output above." >&2
  exit 1
fi
