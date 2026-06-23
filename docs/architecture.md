# isaac-rubato — architecture

How the platform is wired, where the adaptive solver plugs in, and why the GUI and training paths
differ. Companion to the [Home](index.md) and [Roadmap](roadmap.md) pages.

## The stack

```
  Isaac Lab    RL managers · envs · teacher/student · data-gen        (training path)
     │         steps Newton via isaaclab_newton.NewtonMJWarpManager
  Isaac Sim    USD scene · RTX editor · viewer · sensors              (authoring/GUI path)
     │         steps Newton via the isaacsim.physics.newton wheel extensions
  Newton       Model / State + SolverMuJoCo / SolverMuJoCoAdaptive    (the physics component = the fork)
     │
  MuJoCo-Warp  warp-accelerated contact dynamics
```

Three external dependencies, one of them ours:

- **Isaac Sim** (NVIDIA) — pip wheel; provides Kit, the RTX editor, USD, sensors.
- **Isaac Lab** (NVIDIA, BSD-3) — cloned at a pinned commit; the platform applies a small delta.
- **Newton** (Apache-2.0) — the **fork** `newton-adaptive` (a submodule). `811968b` upstream + the
  adaptive-solver commits, including `SolverMuJoCoAdaptive`. Isaac Lab's pinned Newton is overridden
  with this fork (editable), so `import newton` resolves to it everywhere.

## One physics step (where adaptivity lives)

An Isaac Lab env advances by the **frame dt** (`sim.dt`, e.g. 1/120 s). Internally the manager advances
the world over that frame as a sequence of substeps:

```
env.step(action)
  └─ sim advances one frame dt
       └─ NewtonMJWarpManager runs the substep loop
            stock:    solver.step(s_in, s_out, control, contacts, substep_dt)     # FIXED substep
            adaptive: solver.step_dt(substep_dt, s0, s1, control)                 # error-controlled
```

- **Stock (`SolverMuJoCo`)**: each substep is a single fixed integration of `substep_dt`.
- **Adaptive (`SolverMuJoCoAdaptive`)**: `step_dt` runs an **error-controlled inner loop** over
  `substep_dt`. It takes a step, takes two half-steps, compares (**step-doubling**), and accepts or
  subdivides further until the local `joint_q` error is under `tol`. One outer substep therefore becomes
  a **variable** number of inner micro-steps — many under stiff contact, few in free motion. `dt_mode =
  per_world` means each parallel env adapts independently to its own contact state.

**The boundary.** Observations, rewards, actions, rendering, and sensors all sit at or above the frame
dt and are untouched. All adaptivity is *below the frame-dt boundary*, inside the solver. This is why
the integration is a drop-in: nothing above the integrator has to know.

## The seam — the `isaaclab_newton` delta

The delta lives in `isaaclab/` (this repo) and is applied to Isaac Lab's `isaaclab_newton` extension
(`isaaclab/isaaclab_newton_adaptive.patch`). Two source files:

**`mjwarp_manager_cfg.py` — `MJWarpSolverCfg`** gains five fields:

| field | meaning |
|---|---|
| `adaptive` | use `SolverMuJoCoAdaptive` instead of `SolverMuJoCo` |
| `adaptive_tol` | inf-norm `joint_q` error tolerance per world [m or rad] |
| `adaptive_dt_mode` | `per_world` (each env adapts) or `global` (shared worst case) |
| `adaptive_dt_init` | initial inner timestep — set below `sim.dt` to give the controller room |
| `adaptive_dt_min` | inner timestep floor |

**`mjwarp_manager.py` — `NewtonMJWarpManager`**, the four edits:

1. **`_build_solver`** — when adaptive is selected, construct
   `SolverMuJoCoAdaptive(model, tol=…, dt_mode=…, dt_inner_init=…, dt_inner_min=…, **kwargs)`, popping the
   kwargs the adaptive solver sets itself (`use_mujoco_contacts` / `use_mujoco_cpu` / `separate_worlds`).
2. **`_step_solver`** — adaptive → one `step_dt(substep_dt, s0, s1, control)` per substep (the stock
   5-positional `step()` doesn't line up; `step_dt` owns its inner loop and its own contacts).
3. **`_supports_cuda_graph_capture` → False** in adaptive mode — the per-frame substep count is
   data-dependent, so a static CUDA graph can't capture it.
4. **`_log_solver_debug`** — throttled **file** telemetry (Kit swallows stdout): every N frames, per-world
   inner-dt min/max/spread + cumulative substep count to `$NEWTON_ADAPTIVE_LOG` (default
   `/tmp/newton_adaptive.log`).

That is the whole training-path integration: ~5 edits, one file pair.

## Selecting adaptive (three equivalent switches, checked in `_build_solver`)

1. **Config** (the platform path): `MJWarpSolverCfg(adaptive=True, adaptive_dt_init=0.005, …)`.
2. **Env var** (any task, no cfg edit): `NEWTON_ADAPTIVE=1` (+ `NEWTON_ADAPTIVE_TOL`,
   `NEWTON_ADAPTIVE_DTMODE`, `NEWTON_ADAPTIVE_DT_INIT`, `NEWTON_ADAPTIVE_LOG_EVERY`).
3. **GUI carb setting** `/isaaclab/newton/adaptive` — a checkbox (the `newton_adaptive_ui` extension)
   flips it; the solver rebuilds adaptive on the next Stop→Play.

## The two Newton paths (and why the editor differs)

Isaac Lab does **not** use the Isaac Sim wheel's Newton extension; it steps Newton itself through
`isaaclab_newton`. The editor does the opposite. These are genuinely separate code paths:

| | Training path | Editor path |
|---|---|---|
| Entry | `isaaclab_newton.NewtonMJWarpManager` | `isaacsim.physics.newton` (+ `.tensors`, `.ui`) |
| Launched by | `run_native.sh <task>` (Isaac Lab) | `rubato` (`isaacsim.exp.full.newton`) |
| Solver | `SolverMuJoCo` **or `SolverMuJoCoAdaptive`** | `SolverMuJoCo` (stock) |
| Adaptive today | **yes** | not yet (next: config-driven, same solver) |

**Editor-path gotcha (resolved).** The editor's Newton needs three extensions together: the simulator
(`isaacsim.physics.newton`), **the tensors backend (`isaacsim.physics.newton.tensors`)**, and the UI
(`.ui`). Enabling only the simulator registers Newton but leaves
`omni.physics.tensors.create_simulation_view(backend='newton')` with no backend → Play fails with
"Failed to find simulation backend 'newton'". The `isaacsim.exp.full.newton` experience enables all
three; `rubato` launches that experience, which is why Newton now steps in the editor.

## Repository layout

```
isaac-rubato/
  newton-adaptive/   submodule — the Newton fork (SolverMuJoCo / SolverMuJoCoAdaptive)
  isaaclab/          the Isaac Lab delta: isaaclab_newton_adaptive.patch + newton_adaptive_ui/ + INTEGRATION.md
  rl/                scene-agnostic RL tooling
  demos/trossen/     the cube-lift testbed (the contact-rich task adaptive is measured on)
  research/          standalone adaptive evidence (adaptive_expts/) + anymal_study/
  install/           setup.sh + INSTALL.md (assembles Isaac Sim wheel + Isaac Lab@pin + the submodule)
  docs/              this site (index, architecture, components, roadmap, getting-started) + assets/
```

## Evidence

`assets/adaptive_dt_proof.png` — Trossen cube-lift, 64 envs, `dt_mode=per_world`: per-world inner-dt band
on log-y (collapses toward the floor at contact, relaxes to the frame dt in free motion) and
substeps/frame. Measured `dt` range `1.79e-6 → 7.54e-3 s` (**4218×**); **~37 substeps/frame** under
contact (max 42) vs ~3 on a dynamically trivial scene.
