# Adaptive-Solver RL Training — FPS / Performance Test Plan

**Scope:** characterize and increase the "abysmal" training throughput of `SolverMuJoCoAdaptive` vs the fixed-step `SolverMuJoCo` on RL tasks (validation task: franka-reach). This is a planning + diagnosis deliverable. No solver changes implemented here.

**Headline measured result (this session, RTX 4070 Ti SUPER, franka-reach, 1024 envs, seed 42, 6 iters, offline):**

| Solver | Collection time / iter (steady) | Collection FPS (env-steps/s) | Iteration time | Slowdown vs fixed |
|---|---|---|---|---|
| fixed | ~0.131 s | **~188,000** | ~0.19 s | 1× |
| adaptive | ~19.8 s | **~1,240** | ~19.85 s | **~150×** |

`num_steps_per_env=24`, so 1 rsl-rl iteration = 24 × 1024 = 24,576 env-steps. Fixed `Training time` 4.11 s vs adaptive `Training time` 123.13 s. Adaptive substep telemetry (`/tmp/newton_adaptive.log`): **~105–110 MuJoCo opt-steps per frame (~35 boundary iterations × 3 evals)**, with `inner_dt` min repeatedly bottoming at the `1.0e-6` floor while max sits at ~3–5e-3. **Adaptive is ~150× slower than fixed.**

---

## 1. Static bottleneck analysis (code-grounded)

All line refs are in `/home/mdigiorgio/Documents/code/newton-adaptive/newton/_src/solvers/mujoco/solver_mujoco_adaptive.py` unless noted. Manager = `/home/mdigiorgio/Documents/code/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/mjwarp_manager.py`.

### (a) Step-doubling = 3 MuJoCo evals per boundary iteration — ~3× base solve cost
`_run_iteration_body` lines 585–587 run `_run_substep` three times (full dt, half, half). Each `_run_substep` (531–551) is a *full* MuJoCo solve **plus** two state conversions: `_update_mjc_data` (Newton→MJW) and `_update_newton_state` (MJW→Newton), plus a `wp.copy` of `opt.timestep`. So every iteration pays **3 MuJoCo solves + 6 full state conversions**, vs the fixed solver's 1 solve. This is the largest single algorithmic multiplier and is intrinsic to step-doubling error estimation.

### (b) Host sync every boundary iteration — pipeline serialization (the throughput killer)
`step_dt` lines 886–889:
```python
while True:
    self._run_iteration_body(effective_dt_max)
    if self._boundary_flag.numpy()[0] == 0:   # <-- device->host copy = full stream sync
        break
```
`self._boundary_flag.numpy()[0]` forces a **device→host transfer that drains the CUDA stream every iteration**. The GPU cannot run ahead; CPU and GPU ping-pong. With ~18 iterations per `step_dt` and 2 `step_dt` calls per env-step (decimation=2), that is **~36 hard syncs per env-step** — thousands per rsl-rl iteration. At 1024 envs the per-world kernel work is tiny relative to launch+sync latency, so this dominates.

### (c) CUDA-graph capture DISABLED for adaptive — no launch amortization
Manager lines 130–133: `_supports_cuda_graph_capture -> not cls._adaptive`. The **fixed** path captures the entire `decimation × num_substeps` physics loop into **one** graph launched per env-step (`newton_manager.py` `_simulate_full`/`_run_solver_substeps` ~1641–1666). Adaptive issues every kernel eagerly. Per `_run_iteration_body`: **17 explicit `wp.launch` + 4 `wp.copy` + 3 full MuJoCo steps** (each dozens of internal kernels). That is roughly **hundreds of eager kernel launches per env-step** (vs one graph replay), each carrying ~microseconds of launch latency that the graph would erase.

### (d) Data-dependent substep count under stiffness / floor-grinding
`_clamp_dt_to_boundary` (322–341) lands each world on its boundary; `_calc_adjusted_step` (81–159) shrinks dt hard on error and can pin a world at `dt_min=1e-6`. Telemetry shows `inner_dt` min frequently at `1.0e-6` — worlds bottoming out. In **`per_world` mode the `while` loop runs until the *slowest* world reaches the boundary** (`_boundary_check` 282–292 sets the shared flag if *any* world lags), so one stiff world inflates the iteration count — and thus the eval count, sync count, and launch count — **for the entire batch**. `_finish_diverged_worlds` (177–193) caps only NaN/inf worlds, not finite-but-stiff ones.

### (e) Per-iteration kernel-launch + state-select overhead
Beyond the 3 evals, each iteration launches: 2 counters (559–560), clamp (563), 4 state `wp.copy` for rollback (571–576), error kernel (589), optional 3 global-reduce kernels (606–608), Drake controller (610), **4 full-array state-select kernels** over `joint_coord_count`/`joint_dof_count`/`body_count` (630–664), advance-time (666), finish-diverged (675), dt-cap (683), boundary reset+check (691–692). These are all small launches that a captured graph would collapse.

### (f) Per-step `reset()` from Fix C
Manager `_step_solver` lines 113–115 calls `cls._solver.reset(state_0, world_mask=cls._solver.diverged, flags=0)` **every** `step_dt` (i.e. every substep, decimation×num_substeps per env-step). `reset` (931–968) calls `super().reset()` (clears MuJoCo warm-start buffers) **and** launches `_reset_worlds` (953) **unconditionally**, even when zero worlds diverged. This adds a guaranteed per-substep launch + warm-start clear that the fixed path never pays — and clearing warm-start every step also hurts MuJoCo solver convergence (more internal iterations).

### Quantified cost decomposition (matches the measured ~150×)
- Raw solve work: ~108 MuJoCo steps/env-step (adaptive) vs 2 (fixed) = **~54×** from (a)+(d).
- Remaining **~2.8×** on top (54 → measured ~150) = the (b) host-sync + (c) no-graph + (e) eager-launch + per-eval state-conversion + (f) reset tax.

So **~1/3 of the gap (the ~2.8× multiplier) is pure overhead** removable without changing the math; **~2/3 is substep×3-eval algorithmic work** reducible by cutting N and/or the 3rd eval.

---

## 2. Connection to ITEM 4 (even-tiling) — how much FPS it can recover

Even-tiling picks the substep **count N** adaptively from the carried controller dt but sets inner `dt = dt_outer / N` (uniform tiling, exact landing, no ragged remainder). This directly enables the two highest-value FPS levers:

- **Fixed-count inner loop → remove the per-iteration host sync (b).** N is known *before* the loop, so the `while … _boundary_flag.numpy()` (888) becomes a `for k in range(N)` with **zero host syncs inside the interval** (at most one host read per `step_dt` to update N, ideally moved on-device). This alone should remove most of the serialization tax.
- **Static shape → CUDA-graph capturable (c).** With a fixed N the kernel sequence per `step_dt` is static, so `_supports_cuda_graph_capture` could return `True` for adaptive with a **graph cache keyed by N** (recapture only when N changes). Combined with **`global` dt_mode** there is a single shared N for the whole batch (no worst-world domination from (d), no per-world divergence in loop length), which is the clean capturable case.
- **No ragged remainder → fewer tiny trailing iterations** and no floor-grinding from the landing clamp.

**Expected recovery:** even-tiling + global N + graph capture targets essentially all of the **~2.8× overhead multiplier** → from ~150× down to **~50× of fixed** (~1,240 → ~3,700 fps just from overhead removal, holding 3-eval×N fixed). Going further requires attacking the **~54× algorithmic factor**: cap N (Section 4 lever L4), and/or drop the 3rd eval / use a cheaper embedded error estimate (L5). With a capped/normalized N (e.g. ≤8) and graph capture, a realistic target is **within ~10–20× of fixed** (~10–20k fps).

**What blocks it / caveats:**
- N changes between control steps → graph **recapture cost** and cache thrash; needs hysteresis on N (the existing Drake deadband helps) and a small N-keyed cache.
- 3-eval step-doubling is still inside the captured region → the ~3× and the per-eval state conversions remain unless also addressed.
- Fix-C `reset()` (f) launches every step; to stay in one graph it must be folded in or made conditional (skip when no world diverged) — its `diverged` mask read must not introduce a host sync.
- `per_world` even-tiling (N varies per world) is only graph-capturable if you pad the loop to `max(N)` (wastes work on easy worlds) — **global N is the capturable sweet spot**; quantify the accuracy cost of global vs per-world separately (that is ITEM 4's correctness concern, not FPS).

---

## 3. Baseline already measured (this session)

Commands run (one GPU job at a time, both completed cleanly):
```bash
cd /home/mdigiorgio/Documents/code/isaac-rubato/experiments/06-30-2026-experiments
SOLVER=fixed    NUM_ENVS=1024 MAX_ITERATIONS=6 SEED=42 RUN_NAME=fps-fixed \
  WANDB_PROJECT=newton-adaptive-study WANDB_MODE=offline bash franka-reach/train.sh
SOLVER=adaptive NUM_ENVS=1024 MAX_ITERATIONS=6 SEED=42 RUN_NAME=fps-adaptive \
  WANDB_PROJECT=newton-adaptive-study WANDB_MODE=offline NEWTON_ADAPTIVE_LOG_EVERY=10 \
  bash franka-reach/train.sh
```
Results: fixed steady collection **0.131 s/iter (~188k fps)**; adaptive steady collection **19.8 s/iter (~1,240 fps)**; **ratio ~150×**. Substeps/frame ~105–110 (≈35 boundary iters × 3). `inner_dt` min repeatedly at the 1e-6 floor → confirms floor-grinding / stiff-world domination (bottleneck (d)). Learning time is identical (~0.07 s) for both — **the entire gap is in collection (physics)**, confirming the solver, not PPO, is the bottleneck.

---

## 4. The runnable test plan

### 4.1 What to measure (metrics)
1. **FPS = env-steps/sec.** Two splits, both from rsl-rl console: **collection FPS** = `24*NUM_ENVS / Collection_time` (pure physics+env), **overall FPS** = `24*NUM_ENVS / Iteration_time`. Use **steady-state iterations only** (drop iter 0 — it includes CUDA-graph capture for fixed and JIT warm-up for adaptive).
2. **Collection vs learning split** — `Collection time` vs `Learning time` lines. Confirms the gap is physics (it is).
3. **Substeps/frame distribution** — from `/tmp/newton_adaptive.log`: `Δcumulative_substeps / Δframe`, and `inner_dt[min,max,spread]`. Report mean, p95, max. min==1e-6 ⇒ floor-grinding.
4. **Host-sync count/step** — = boundary iterations/step = `cumulative_substeps/(3·frames)`. This is the `.numpy()` count and the primary serialization metric.
5. **GPU utilization** — `nvidia-smi dmon` during steady iters; low util on adaptive confirms host-sync stalls (bottleneck (b)).
6. **Wall-clock** — `Training time` line as a cross-check on per-iter sums.

### 4.2 Controlled harness (exact commands)
Fix everything except the one lever: `NUM_ENVS=1024 MAX_ITERATIONS=8 SEED=42 WANDB_MODE=offline` on franka-reach. Use ≥8 iters, average iters 2–7 (warm-up = iter 0–1). **Never run two GPU jobs concurrently** (one GPU). Capture GPU util in parallel with the run, not a second training job:
```bash
cd /home/mdigiorgio/Documents/code/isaac-rubato/experiments/06-30-2026-experiments
EXP=franka-reach
run() {  # $1=label  $2..=extra env
  local L=$1; shift
  nvidia-smi dmon -s u -d 1 -o T > /tmp/fps_${L}_gpu.log &  SMID=$!
  /usr/bin/time -v env "$@" SOLVER=${SOLVER:-adaptive} NUM_ENVS=1024 MAX_ITERATIONS=8 SEED=42 \
    RUN_NAME=fps-$L WANDB_PROJECT=newton-adaptive-study WANDB_MODE=offline \
    NEWTON_ADAPTIVE_LOG_EVERY=10 bash $EXP/train.sh > /tmp/fps_${L}.log 2>&1
  kill $SMID 2>/dev/null
  echo "== $L =="; grep -E "Collection time|Iteration time|Learning time" /tmp/fps_${L}.log | sed -n '3,8p'
  awk '/cumulative_substeps/{n=$NF} END{print "last cumulative_substeps="n}' /tmp/newton_adaptive.log
}
# Baselines
SOLVER=fixed    run fixed
SOLVER=adaptive run adaptive_perworld   NEWTON_ADAPTIVE_DTMODE=per_world
SOLVER=adaptive run adaptive_global     NEWTON_ADAPTIVE_DTMODE=global
```
Read FPS: `fps = 24*1024 / mean(Collection time over iters 2..7)`. Read substeps/frame from `/tmp/newton_adaptive.log` deltas.

### 4.3 Levers to test, predicted impact (ranked)

| # | Lever | How to test (today, no code) | Predicted FPS impact | Bottleneck targeted |
|---|---|---|---|---|
| **L1** | **even-tiling + global N + CUDA-graph capture** | Requires ITEM 4 code; until then proxy = `NEWTON_ADAPTIVE_DTMODE=global` to measure global-mode substep reduction, and inspect graph-capture path | **Largest.** Removes ~2.8× overhead; ~150×→~50×; with capped N → ~10–20× | (b) sync, (c) graph, (e) launches, (d) worst-world |
| **L2** | **Remove/sparsify host sync** (fixed-count loop, or sync every K iters) | ITEM 4 fixed-count loop; or proxy: measure host-sync count via metric #4 and correlate with GPU util #5 | High — removes serialization; recovers much of the ~2.8× | (b) |
| **L3** | **`global` vs `per_world` dt_mode** | `NEWTON_ADAPTIVE_DTMODE=global` (env var, runs now) | Medium — kills worst-world domination of loop length; measure Δsubsteps/frame | (d) |
| **L4** | **Cap max substeps/frame** (raise `dt_inner_min`, set `dt_inner_max`) | `NEWTON_ADAPTIVE_DT_MIN=1e-4 NEWTON_ADAPTIVE_DT_INIT=...` (runs now); watch divergence/accuracy | Medium-high — directly cuts N; trades accuracy | (a),(d) |
| **L5** | **Skip 3rd eval / cheaper error estimate** | Requires code (embedded estimate) | ~up to 1.5× (3 evals→2) | (a) |
| **L6** | **Make Fix-C `reset()` conditional** (skip when `diverged` all-False) | Requires code; cheap | Small but per-step | (f) |
| **L7** | **Fuse state-select / counter kernels** | Requires code | Small (folded by graph anyway) | (e) |

Run L3 and L4 **today** (env-var only) to get real numbers before any ITEM 4 code lands; they isolate the algorithmic (substep-count) half of the gap. L1/L2 require the even-tiling implementation.

### 4.4 Tuning-sweep (env-var only, runnable now)
```bash
for DM in per_world global; do
  for DTMIN in 1e-6 1e-5 1e-4; do
    SOLVER=adaptive NEWTON_ADAPTIVE_DTMODE=$DM NEWTON_ADAPTIVE_DT_MIN=$DTMIN \
      NUM_ENVS=1024 MAX_ITERATIONS=8 SEED=42 RUN_NAME=fps-$DM-$DTMIN \
      WANDB_PROJECT=newton-adaptive-study WANDB_MODE=offline NEWTON_ADAPTIVE_LOG_EVERY=10 \
      bash franka-reach/train.sh > /tmp/fps_${DM}_${DTMIN}.log 2>&1
    echo "$DM dtmin=$DTMIN: collection=$(grep 'Collection time' /tmp/fps_${DM}_${DTMIN}.log | sed -n '4p')"
  done
done
```
For each: record collection FPS, mean/p95/max substeps/frame, and any `diverged`/accuracy regression (check `Metrics/ee_pose/position_error` doesn't blow up). This maps the **FPS-vs-accuracy frontier** of the substep-count lever.

### 4.5 Warm-up / hygiene
- Discard iteration 0 always (graph capture / JIT). Average iters 2–7.
- `rm -f /tmp/newton_adaptive.log` before each adaptive run (train.sh already does for `SOLVER=adaptive`).
- One GPU job at a time. Run `nvidia-smi dmon` as a *monitor*, never a second train.
- Keep `WANDB_MODE=offline` to avoid network jitter in timings.
- Same `NUM_ENVS`/`SEED`/task across all cells so only the lever varies.

### 4.6 Pass/fail targets
- **P0 (diagnosis, met this session):** quantified fixed:adaptive ratio (~150×) + substeps/frame (~108) + collection-dominated gap. ✅
- **P1 (overhead removed):** after L1+L2 (even-tiling global + graph + no per-iter sync), **adaptive collection FPS ≥ fixed/50** (≥ ~3,700 fps at this config) with host-sync count/step → ≤1.
- **P2 (usable):** with L1+L4 (capped global N ≤ 8), **adaptive within 10–20× of fixed** (≥ ~10k fps) **and** no accuracy regression vs current adaptive (`ee_pose/position_error` within noise, `diverged` rate unchanged).
- **P3 (stretch):** GPU util during adaptive collection ≥ 70% (currently expected low due to host-sync stalls), confirming the pipeline no longer serializes.

---

## Files
- Solver under analysis: `/home/mdigiorgio/Documents/code/newton-adaptive/newton/_src/solvers/mujoco/solver_mujoco_adaptive.py` (hot loop `_run_iteration_body` 553–697; host sync `step_dt` 886–889; 3-eval 585–587).
- Manager: `/home/mdigiorgio/Documents/code/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/mjwarp_manager.py` (graph-disable 130–133; per-step reset 113–115).
- Train script: `/home/mdigiorgio/Documents/code/isaac-rubato/experiments/06-30-2026-experiments/franka-reach/train.sh`.
- Baseline logs (this session): `/tmp/claude-1000/-home-mdigiorgio-Documents-code-isaac-rubato/e383142e-b33e-4fa9-86bf-08bb5c1fc2f6/scratchpad/fixed.log` and `adaptive.log`; substep telemetry `/tmp/newton_adaptive.log`.