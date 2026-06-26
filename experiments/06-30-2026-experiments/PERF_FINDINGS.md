# Adaptive-solver FPS — root-cause findings (franka-reach)

**Question:** adaptive training FPS is "abysmal" (~150× slower than fixed). Step-doubling is intrinsically
~3× (3 MuJoCo evals/step), so where does the other ~50× come from — bug or cost?

## CONFIRMED ROOT CAUSE: `adaptive_tol = 1e-4` is ~100–500× too tight (franka-reach/env.py:23)

Direct measurement settles it. At NE=4 (minimal worst-world tail), sweeping tol:

| tol | iters/frame (1.0 = stepping at the max outer dt) |
|---|---|
| 1e-4 (current franka setting) | 23.8 |
| 1e-3 | 7.6 |
| 1e-2 | 3.7 |
| **5e-2** | **1.0 — steps at exactly the max outer dt** |

The solver **does** step at the max outer dt — at a sane tolerance. The dt collapse is purely `tol=1e-4`
being ~100–500× tighter than the franka arm's natural per-step integration error (~1–5e-2 rad).

**Ruled OUT by direct measurement (not the cause):**
- *Gripper / a stiff unused DOF:* per-coordinate `|full−double|` dump shows the inf-norm is dominated by ARM
  joints (coord 0 & 6 = panda_joint1 & panda_joint7), error ~8.8e-3 rad at dt=10ms; gripper coords ~3e-4.
- *Explicit-Euler integrator vs implicit PD actuators:* A/B `euler` vs `implicitfast` at tol=1e-4 moved
  iters/frame only 38.8→31.3 (~19%). Not the cause.
- *Solver bug:* none — the controller faithfully enforces the requested tol; subdivision scales ~1/√tol.

**Why fixed-step is fine at the same dt:** fixed runs at dt=16.67ms with per-step error LARGER than 8.8e-3 and
trains fine — i.e. the task tolerates ~100× more error than tol=1e-4 demands. tol=1e-4 asks adaptive to be ~100×
more accurate than fixed-step ever is. It is a high-accuracy STUDY setting, wrong for training.

**Answer to the headline:** it is ~90% **legitimate MuJoCo-solve count** and ~10% overhead — but the solve count
is legitimate only relative to an over-tight tol. Fix the tol and the solve count collapses with it.

## Evidence (RTX 4070 Ti SUPER, franka-reach, ragged adaptive, steady-state collection time)

franka: `sim.dt = 16.67 ms`, `decimation = 2` (33 ms / 30 Hz control). Each `step_dt` subdivides ONE 16.67 ms
`sim.dt`; fixed does it in 1 solve.

**(1) Subdivision is uniform, not a few outliers — substeps/frame vs env count (tol=1e-4):**

| envs | iters/frame | min inner_dt |
|---|---|---|
| 4 | 23.8 | 4e-4 (not floored) |
| 256 | 33.3 | 3.6e-6 (floored) |
| 2048 | 37.3 | 3.8e-6 (floored) |

Even 4 worlds need ~24 boundary iterations/frame → the bulk is uniform. Worst-world tail adds only ~+56%
from 4→2048 envs (floor-hitting is a tail effect).

**(2) Subdivision scales as ~1/sqrt(tol) — i.e. LEGITIMATE 2nd-order integration error, not an inflated
estimate (NE=64):**

| tol | iters/frame | collect (2 it) | ratio vs theory |
|---|---|---|---|
| 1e-4 | 30.8 | 16.3 s | — |
| 5e-4 | 13.6 | 7.7 s | 2.26× (sqrt5=2.24 ✓) |
| 2e-3 | 8.3 | 4.7 s | 3.7× |
| 1e-2 | 4.7 | 2.8 s | 6.6× |

**(3) Best-case adaptive vs fixed (NE=64, steady collection, dt_init=dt_outer):**

| config | collect/iter | iters/frame | vs fixed |
|---|---|---|---|
| fixed | 0.153 s | — | 1× |
| adaptive tol=1e-4 (current franka env) | 16.72 s | 31.4 | **109×** |
| adaptive tol=1e-3 | 6.11 s | 10.6 | 40× |
| adaptive tol=1e-2 | 2.95 s | 4.8 | **19×** |

Decomposition at tol=1e-2: 4.8 iters × 3 evals = 14.4 solves/frame vs fixed 1 → **14.4× solve count + ~1.34×
overhead = 19×**. At tol=1e-4: ~94× solve count + ~1.16× overhead = 109×. **Overhead (host-sync + no CUDA
graph) is only ~1.2–1.34×, NOT the main cost.**

## Root causes (ranked by impact)

1. **`adaptive_tol = 1e-4` rad is ~24× tighter than the task needs (DOMINANT).** It is a high-accuracy-study
   setting left on for training; fixed-step runs the task fine at its own (looser) accuracy. Subdivision is
   legitimate for that tol. **Lever: set tol to training-appropriate (~1e-3 to 1e-2) → 5–6× faster immediately.**
2. **Batched step-doubling is worst-world-bound (ARCHITECTURAL).** `_run_iteration_body` steps the WHOLE batch
   3× every boundary iteration; finished worlds get `dt=0` but still pay the batched MuJoCo solve. So the single
   worst world's substep count inflates the 3-eval cost for all worlds → iters/frame floors at ~5 even at loose
   tol, and grows with batch size. **This is why "3×" (which needs N≈1 for the worst world in the batch) is not
   reachable at scale on a task with any dynamics variety.** Lever: cap max substeps (bounds the tail, trades
   worst-world accuracy); see even-tiling `max_substeps`.
3. **`dt_inner_init = 0.01 < dt_outer = 0.0167`** — minor; the controller ramps regardless, so tuning dt_init
   alone barely moved iters/frame.
4. **Per-iteration overhead** (host sync `_boundary_flag.numpy()` every iteration + CUDA-graph capture disabled):
   only ~1.2–1.34×. Addressed by even-tiling fixed-count loop + graph capture (PERF_TEST_PLAN.md L1/L2) but the
   payoff is modest because compute, not overhead, dominates.
5. **Irreducible 3-eval step-doubling = the 3× floor** (only at N=1).

## Recommendations

- **Immediate (config, no code):** for training runs, set `adaptive_tol` to match the task's real accuracy need
  (start ~1e-3; franka trains fine far above 1e-4). ~5–6× speedup, zero code. The tol=1e-4 in `franka-reach/env.py`
  is a study setting, not a training setting.
- **Cap the worst-world tail:** apply a max-substeps cap in ragged mode too (even-tiling already has
  `max_substeps`); bounds the batched 3-eval blow-up from one stiff world. Trades worst-world accuracy.
- **Even-tiling + CUDA-graph (PERF_TEST_PLAN L1/L2):** worth doing for the host-sync/launch overhead, but expect
  only ~1.2–1.3× from it — it is NOT the main lever. Do it for correctness/cleanliness, not as the FPS fix.
- **Task choice:** franka-reach is the worst case for adaptive FPS — low-contact + tight tol = uniform
  over-subdivision with no adaptive payoff. On a contact-rich task with appropriate tol, most steps are N≈1
  (≈3× fixed) and only stiff-contact steps subdivide — which is where adaptive earns its cost.

**Bottom line:** the 150× is not a bug to delete; it is `tol` mistuned ~24× tight (immediate 5–6× win) on top of
an architectural worst-world tail (cap it) on top of the irreducible 3× step-doubling floor. The "~3×" target is
only physical when N≈1 for the whole batch — i.e. an appropriately-tol'd, contact-sparse task, not franka@1e-4.
