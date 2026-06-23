# Components

isaac-rubato is the umbrella. Isaac Sim and Isaac Lab are upstream dependencies (not vendored); Newton
is a maintained fork, included as a submodule. The platform's own code — the integration, the RL
tooling, the scenes, and the install automation — lives in this repository.

```
isaac-rubato/
  newton-adaptive/   submodule — the Newton fork (SolverMuJoCo / SolverMuJoCoAdaptive)
  isaaclab/          the Isaac Lab delta: patch + GUI-toggle extension + INTEGRATION.md
  rl/                scene-agnostic RL tooling
  demos/trossen/     the cube-lift testbed
  research/          standalone adaptive evidence + studies
  install/           setup.sh + INSTALL.md
  docs/              this site
```

## newton-adaptive — the physics component

The Newton fork. `811968b` upstream plus the adaptive-solver commits, including **`SolverMuJoCoAdaptive`**
— error-controlled step-doubling over MuJoCo-Warp. Isaac Lab's pinned Newton is overridden with this
fork (editable), so `import newton` resolves to it across the whole stack.

Newton is **one component** of the platform, not its root.

## isaaclab — the integration delta

The small set of edits that teach Isaac Lab's Newton backend to use the adaptive solver, kept as a patch
plus a Kit extension so they re-apply cleanly onto a fresh Isaac Lab clone:

- `isaaclab_newton_adaptive.patch` — the `MJWarpSolverCfg` fields and the `NewtonMJWarpManager` seam.
- `newton_adaptive_ui/` — a Kit extension that adds an **Adaptive timestepping** checkbox to the editor
  (drives the `/isaaclab/newton/adaptive` carb setting).
- `INTEGRATION.md` — the authoritative description of the seam and how to re-apply it.

See [Architecture](architecture.md) for the seam itself.

## rl — scene-agnostic RL

Reusable training, launching, backend selection, and evaluation tooling — the parts shared across
scenes. (As the second scene lands, the general pieces are factored out of `demos/` into here.)

## demos — scenes

Concrete tasks the platform runs. Today: **`trossen/`**, a Trossen Stationary-AI cube-lift — a
contact-rich manipulation task and the one the adaptive solver is measured on. Planned: a Unitree G1
and a Leap hand, to exercise locomotion and dexterous contact.

## research — evidence

Standalone studies that don't need the full stack:

- `adaptive_expts/` — adaptive-vs-fixed work-precision and stiff-contact experiments at the Newton level.
- `anymal_study/` — a completed reference study.

## install — assembly

`setup.sh` reconstructs the full environment on a fresh machine: the Isaac Sim wheel, Isaac Lab at a
pinned commit with the delta applied, and the Newton submodule wired in. See
[Getting started](getting-started.md).
