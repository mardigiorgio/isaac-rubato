# franka-reach

Franka arm tracking an end-effector pose target. Low-contact (like cartpole), so expect reward
parity fixed-vs-adaptive - it's a manipulation warm-up, not a contact showcase.

```
env.py        stock franka-reach + Newton njmax fix, registered as Isaac-Reach-Franka-Rubato
train.sh      SOLVER=fixed|adaptive [VIDEO=1] bash train.sh
scenes/       GUI-built .usd scenes for this experiment (see scenes/README.md)
```

Run:
```bash
SOLVER=fixed    VIDEO=1 bash train.sh
SOLVER=adaptive VIDEO=1 bash train.sh
```

The `njmax` bump (50 -> 128) in `env.py` fixes the stock buffer overflow on the Newton solver
(`nefc overflow ... increase njmax`). Goal marker in videos is still a known gap (see top-level README).
