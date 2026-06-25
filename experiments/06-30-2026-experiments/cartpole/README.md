# cartpole - experiment #1 (the smoke)

Trivial control, no stiff contact. It exists to prove the adaptive Newton path trains a competent
policy cleanly before spending compute on hard tasks. Read the result as **reward parity**: fixed
and adaptive should learn the same curve.

Note: even here the adaptive controller subdivides (~12-37 substeps/frame, inner-dt spread
~4-8e-3 s) because the default `dt_inner_init=0.01` is larger than the per-frame substep - that's
config-driven subdivision, not stiff-contact adaptation. The contact-rich payoff shows up on
reorient/stack, not here.

```
env.py        stock Isaac-Cartpole, registered as Isaac-Cartpole-Rubato (edit-home; no overrides yet)
train.sh      SOLVER=fixed|adaptive [VIDEO=1] bash train.sh
validate.sh   keyless smoke (64 envs, 50 iters, offline) -> proves the adaptive path trains
```

Run:
```bash
SOLVER=fixed    bash train.sh
SOLVER=adaptive bash train.sh
bash validate.sh        # keyless
```
