# Adaptive Newton backend - Isaac Lab integration (PROVEN, 2026-06)

`SolverMuJoCoAdaptive` (error-controlled step-doubling) runs as a **selectable solver mode** of Isaac Lab's
Newton backend and **visibly adapts** on contact-rich manipulation: it subdivides time where contact is
stiff, while trivial tasks stay at the frame boundary.

## The integration point - `isaaclab_newton`, NOT the wheel

Isaac Lab's `env.sim.physics=newton_mjwarp` is stepped by the **native** `isaaclab_newton.NewtonMJWarpManager`
- it builds the solver and steps it itself, explicitly bypassing the Isaac Sim wheel extension
`isaacsim.physics.newton` ("no isaacsim.physics.newton extension needed", `newton_manager.py:392`). So the
integration is **5 edits to that manager + its cfg** (editable Isaac Lab source = the "modified Isaac Lab"),
captured in [`isaaclab_newton_adaptive.patch`](isaaclab_newton_adaptive.patch):

1. `MJWarpSolverCfg`: `adaptive` + `adaptive_tol`/`adaptive_dt_mode`/`adaptive_dt_init`/`adaptive_dt_min` fields.
2. `NewtonMJWarpManager._build_solver`: when `adaptive`, construct `SolverMuJoCoAdaptive` (popping the kwargs it
   forces - `use_mujoco_contacts`/`use_mujoco_cpu`/`separate_worlds`) instead of stock `SolverMuJoCo`.
3. `NewtonMJWarpManager._step_solver`: adaptive → `step_dt(substep_dt, s0, s1, control)` once per substep (the
   stock 5-positional `step()` is incompatible; `step_dt` owns the inner error-controlled loop + its contacts).
4. `_supports_cuda_graph_capture` → `False` when adaptive (the per-frame substep count is data-dependent).
5. `_log_solver_debug` → throttled **file** telemetry (`$NEWTON_ADAPTIVE_LOG`, default `/tmp/newton_adaptive.log`)
   - Kit swallows stdout, so the dt/substep proof must go to a file.

Re-apply on a fresh Isaac Lab clone: `bash integration/apply_isaaclab_delta.sh ../IsaacLab` (idempotent; the
install runs this).

## Selecting adaptive
`_build_solver` builds `SolverMuJoCoAdaptive` when **any** of these is set (checked in this order):
- **Config-driven** (the platform path): `NewtonCfg(solver_cfg=MJWarpSolverCfg(adaptive=True, adaptive_dt_init=0.005, …))`.
  A task's `newton_mjwarp` preset can bake this in, so the task runs adaptive by default.
- **Env toggle** (any task, no cfg edit): `NEWTON_ADAPTIVE=1`; tune with `NEWTON_ADAPTIVE_DT_INIT`,
  `NEWTON_ADAPTIVE_DTMODE` (`per_world`|`global`), `NEWTON_ADAPTIVE_TOL`, `NEWTON_ADAPTIVE_LOG_EVERY`.
- **GUI toggle** (carb setting `/isaaclab/newton/adaptive`): a checkbox in the editor. Flip it, then
  **Stop → Play** (the toggle applies at solver-build time - switching the integrator rebuilds the solver;
  a live mid-sim swap is intentionally avoided). Confirm via `/tmp/newton_adaptive.log` (spread > 0 +
  many substeps = on). Two ways to get the checkbox:
  - **Autorun (default):** the Kit extension `IsaacLab/source/newton_adaptive_ui/` is listed in
    `apps/isaaclab.python.kit` `[dependencies]` (`"newton_adaptive_ui" = {order = 1001}`), so a
    **Newton Integrator** window appears automatically in every Isaac Lab **GUI** session. It is *not* in
    the headless kits (`isaaclab.python.headless*.kit` define their own deps), so headless training never
    loads the UI. The rendering GUI variant inherits it via its `"isaaclab.python"` dependency.
  - **No-install fallback:** paste [`gui_toggle.py`](gui_toggle.py) into the Script Editor (Window >
    Script Editor > Run) - same window, same carb setting, for a stock Isaac Lab clone without the extension.

## What "adapting" looks like

The per-frame dt/substep telemetry (`/tmp/newton_adaptive.log`, written when `NEWTON_ADAPTIVE_LOG_EVERY=N`)
shows the controller spending inner steps where contact is stiff and withholding them in free motion:

| task | inner-dt spread | substeps/frame |
|---|---|---|
| trivial control (cartpole) | ~flat, pinned near the frame boundary | low |
| contact-rich manipulation | wide: `~1e-4` (stiff contact) → `~5e-3` (free motion) | many |

The contact-rich payoff is measured by the fixed-vs-adaptive runs in
`experiments/06-30-2026-experiments/` (cube-reorient, stack) - exactly where stiff contact dominates.

## Verify it yourself
```bash
# any task on the adaptive backend writes per-frame dt/substep telemetry to a fresh log:
rm -f /tmp/newton_adaptive.log
cd experiments/06-30-2026-experiments
bash cartpole/validate.sh        # keyless: proves the solver builds, steps, and logs
tail /tmp/newton_adaptive.log    # spread + substeps/frame
```
On a contact-rich task (cube-reorient, stack) the spread widens and substeps/frame climb where contact
is stiff - that's the thesis. Drop `NEWTON_ADAPTIVE` for the stock-Newton baseline.

## GUI control: a checkbox, not the engine dropdown
Adaptive is a **solver mode** of the Newton backend, not a separate physics engine, so the right GUI affordance
is a **toggle** (`gui_toggle.py`, above), not a new entry in the `omni.physics` engine dropdown. The dropdown
lists engines (PhysX / Newton); the checkbox flips the active Newton solver between fixed-step and adaptive.
