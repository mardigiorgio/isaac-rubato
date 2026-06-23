# isaac-rubato

A research platform for adaptive-physics reinforcement learning, built on Isaac Sim, Isaac Lab, and
Newton.

*Rubato* (It., "stolen"): in musical performance, the flexible variation of local tempo — time taken
from some beats and repaid in others. The term is used here for adaptive timestepping: integration
effort is reallocated across a frame, spent on stiff-contact intervals and withheld from free motion,
while the frame boundary is preserved.

## Motivation

Physics-based RL integrates contact dynamics at a fixed substep. A fixed step is a single compromise
applied uniformly: large enough to be efficient in free motion, it under-resolves stiff or
near-rigid contact, admitting integration error, penetration, and instability; small enough to resolve
contact, it over-resolves the rest and wastes computation. Contact-rich manipulation — the regime of
interest — is precisely where this compromise is worst, and where simulation error degrades both
sample efficiency and sim-to-real transfer.

Two error sources are separable. The first is temporal: the integrator's step relative to the local
stiffness. The second is the contact model itself: non-convex, iterative contact solvers introduce
their own error and non-smoothness. Adaptive timestepping addresses the first by controlling local
integration error per substep. Convex contact integration (ICF / SAP) addresses the second.

## Goal

The platform currently provides **adaptive timestepping** through `SolverMuJoCoAdaptive`
(error-controlled step-doubling over MuJoCo-Warp), exposed as a selectable solver in Isaac Lab's
Newton backend. The research target is **CENIC**: adaptive timestepping combined with convex contact
integration. "Adaptive" refers to the current solver; "CENIC" is reserved for the adaptive-plus-convex
method.

The platform integrates this solver into a standard RL stack (Isaac Sim, Isaac Lab) so that the
adaptive backend can be selected, trained against, and evaluated on contact-rich tasks without
modifying the layers above the integrator.

## Components

Isaac Sim and Isaac Lab are upstream dependencies, not vendored here; Newton is a maintained fork and
is included as a submodule.

| Path | Component | Role |
|---|---|---|
| `newton-adaptive/` | Newton fork (submodule) | physics engine and `SolverMuJoCoAdaptive` |
| `isaaclab/` | Isaac Lab delta | patch and GUI extension wiring adaptive into Isaac Lab's Newton backend |
| `rl/` | general RL | scene-agnostic launcher, training harness, backend selection, evaluation |
| `demos/` | scenes | `trossen/` cube-lift testbed; further scenes to follow |
| `research/` | evidence | `adaptive_expts/` work-precision studies; `anymal_study/` |
| `install/` | install automation | assembles Isaac Sim, Isaac Lab at a pinned commit, and the Newton submodule |

## Layout

```
isaac-rubato/
  newton-adaptive/   submodule -> github.com/mardigiorgio/newton-adaptive  (Newton + adaptive solver)
  isaaclab/          Isaac Lab adaptive delta (patch, GUI-toggle extension, INTEGRATION.md)
  rl/                scene-agnostic RL tooling
  demos/trossen/     cube-lift testbed
  research/          standalone adaptive evidence and studies
  install/           setup.sh, INSTALL.md
```

## Quickstart

```bash
git clone --recurse-submodules https://github.com/mardigiorgio/isaac-rubato.git
cd isaac-rubato
bash install/setup.sh    # prerequisites in install/INSTALL.md (Ubuntu, NVIDIA driver >= 580, ~60 GB)
```

Open the interactive editor on the Newton backend (drop an object → Play → it falls on Newton):

```bash
rubato                   # launches the isaacsim.exp.full.newton experience
```

## Docs

- [`docs/PI_BRIEF.md`](docs/PI_BRIEF.md) — one-page briefing: what works, the seam, the demo, Q&A, the CENIC roadmap.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how it's wired: the two Newton paths, the `isaaclab_newton` seam, selecting adaptive.
- [`docs/adaptive_dt_proof.png`](docs/adaptive_dt_proof.png) — adaptive dt subdividing at contact on the Trossen task.

## Attribution

Built on NVIDIA Isaac Sim and Isaac Lab (BSD-3-Clause) and the Newton physics engine (Apache-2.0),
which build on MuJoCo and MuJoCo-Warp. These are upstream projects under their own licenses. This
repository contains the adaptive-solver integration, RL workstream, scenes, and install automation.
