# isaac-rubato

isaac-rubato integrates an error-controlled, adaptive-timestepping physics solver as a *selectable*
backend within a standard reinforcement-learning stack — [Isaac Sim](https://developer.nvidia.com/isaac/sim),
[Isaac Lab](https://github.com/isaac-sim/IsaacLab), and [Newton](https://github.com/newton-physics/newton) —
so that fixed- and adaptive-step integration can be selected, trained against, and compared on
contact-rich manipulation tasks without modifying the layers above the integrator. It is a step toward
**CENIC**: adaptive timestepping combined with convex contact integration.

*Rubato* (It., "stolen"): the flexible variation of local tempo in musical performance — time taken from
some beats and repaid in others. Used here for adaptive timestepping, where integration effort is
reallocated across a frame, spent on stiff-contact intervals and withheld from free motion, while the
frame — and with it the fixed-rate control boundary — is preserved exactly.

## Motivation

Physics-based RL integrates contact dynamics at a fixed sub-step: a single compromise applied uniformly.
Coarse enough to be efficient in free motion, it under-resolves stiff contact — integration error,
penetration, instability; fine enough to resolve contact, it over-resolves the rest. Contact-rich
manipulation is where this compromise is worst, and where simulation error most degrades sample
efficiency and sim-to-real transfer.

Two error sources are separable: the *temporal* error (the integrator's step relative to local
stiffness) and the *contact-model* error (non-convex, iterative contact solvers). Adaptive timestepping
addresses the first; convex contact integration (ICF / SAP) addresses the second. Their combination is
CENIC.

## Integration

Isaac Lab advances physics through a single manager class (`NewtonMJWarpManager`, in its
`isaaclab_newton` extension) that builds and steps the Newton solver each control interval. The
integration is localized to two methods on that class — solver construction and solver stepping — so the
adaptive solver (`SolverMuJoCoAdaptive`, error-controlled step-doubling over MuJoCo-Warp) is selectable
alongside PhysX and stock Newton, with the policy, rewards, observations, and rendering unchanged. The
control interface is preserved exactly: the action is zero-order-held across the interval, and the solver
subdivides time only within the interval, always landing on the boundary. See
[`docs/architecture.md`](docs/architecture.md).

## Components

| Path | Component | Role |
|---|---|---|
| `newton-adaptive/` | Newton fork | physics engine + `SolverMuJoCoAdaptive` (cloned at install time) |
| `isaaclab/` | Isaac Lab delta | the patch + Kit extension that wire adaptive into Isaac Lab's Newton backend |
| `rl/` | general RL | scene-agnostic launcher, training harness, backend selection, evaluation |
| `demos/` | scenes | `trossen/` cube-lift testbed; further scenes to follow |
| `research/` | evidence | `adaptive_expts/` work-precision studies; `anymal_study/` |
| `install/` | install automation | assembles Isaac Sim, Isaac Lab (pinned), and the Newton fork |

## Quickstart

```bash
git clone https://github.com/mardigiorgio/isaac-rubato.git
cd isaac-rubato
bash install/setup.sh    # prerequisites in install/INSTALL.md (Ubuntu, NVIDIA driver >= 580, ~60 GB)
```

`setup.sh` installs the Isaac Sim 6.0 wheel, clones Isaac Lab at a pinned commit and the Newton fork,
applies the adaptive delta, and verifies the solver. Open the interactive editor on the Newton backend:

```bash
rubato                   # launches the isaacsim.exp.full.newton experience
```

## Documentation

A browsable site (MkDocs Material) — serve it locally with `rubato-docs` (→ <http://127.0.0.1:8000>),
or read the sources:

- [`docs/architecture.md`](docs/architecture.md) — the stack, the two Newton paths, the integration seam, selecting adaptive.
- [`docs/components.md`](docs/components.md) — the platform's parts.
- [`docs/roadmap.md`](docs/roadmap.md) — adaptive → CENIC.
- [`docs/getting-started.md`](docs/getting-started.md) — install, run, select the solver.

## Built on

This platform is layered on three upstream projects, used under their own licenses:

- **Isaac Sim** — NVIDIA Omniverse robotics simulator (Kit, RTX renderer, USD). <https://developer.nvidia.com/isaac/sim>
- **Isaac Lab** — NVIDIA's robot-learning / RL framework on Isaac Sim (BSD-3-Clause). <https://github.com/isaac-sim/IsaacLab>
- **Newton** — GPU physics engine over MuJoCo-Warp (Apache-2.0). <https://github.com/newton-physics/newton>

which in turn build on **MuJoCo** and **MuJoCo-Warp** (Google DeepMind). This repository contributes the
adaptive-solver integration, the RL workstream, the scenes, and the install automation; the upstream
projects above remain under their respective licenses and copyrights.
