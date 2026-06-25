"""NaN-localization probe for the Shadow Hand reorient task on the Newton backend.

Subclasses the stock ReorientDirectEnv and, the first time the policy observation goes
non-finite, dumps WHICH underlying tensor is bad to /tmp/shadow_nan_probe.txt and halts.
This separates the two solver-independent causes:
  (A) degenerate joint limits   -> scale_transform(dof_pos) NaN, dof obs bad, limits show upper==lower
  (B) bad initial state/contact -> object_*/fingertip_* physics state NaN after one step

Use via train.sh:
  NUM_ENVS=16 MAX_ITERATIONS=1 SOLVER=fixed \
  REGISTER=debug_env.register TASK=Isaac-Reorient-Cube-Shadow-Debug \
  bash shadow-hand-repose-cube/train.sh
"""

import os

import gymnasium as gym
import torch
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg
from isaaclab_tasks.core.reorient.reorient_direct_env import ReorientDirectEnv

PROBE_OUT = os.environ.get("SHADOW_NAN_PROBE", "/tmp/shadow_nan_probe.txt")
TASK_ID = "Isaac-Reorient-Cube-Shadow-Debug"


def _finite(t: torch.Tensor) -> str:
    if t is None:
        return "None"
    n_nan = int(torch.isnan(t).sum())
    n_inf = int(torch.isinf(t).sum())
    flag = "OK" if (n_nan == 0 and n_inf == 0) else f"BAD nan={n_nan} inf={n_inf}"
    return f"{flag} shape={tuple(t.shape)} min={t.float().min():.4g} max={t.float().max():.4g}"


class DebugReorientEnv(ReorientDirectEnv):
    _probe_done = False
    _obs_calls = 0

    def _dump(self, tag: str) -> None:
        h = self.hand.data
        o = self.object.data
        lo = self.hand_dof_lower_limits
        hi = self.hand_dof_upper_limits
        width = (hi - lo)
        degenerate = (width.abs() < 1e-9)
        lines = [
            f"=== shadow_hand NaN probe ({tag}) ===",
            f"num_envs={self.num_envs} num_hand_dofs={self.num_hand_dofs}",
            "",
            "-- joint limits (cause A: scale_transform divides by upper-lower) --",
            f"lower      : {_finite(lo)}",
            f"upper      : {_finite(hi)}",
            f"width(hi-lo): {_finite(width)}",
            f"degenerate joints (width<1e-9): {int(degenerate.sum())} of {lo.numel()}",
        ]
        if int(degenerate.sum()) > 0:
            # joint_names aligns with the per-dof axis
            names = getattr(self.hand, "joint_names", None)
            bad_axes = torch.unique(degenerate.nonzero()[:, -1]).tolist()
            lines.append(f"  degenerate dof indices: {bad_axes}")
            if names:
                lines.append(f"  degenerate dof names  : {[names[i] for i in bad_axes if i < len(names)]}")
        tensors = {
            "hand joint_pos": h.joint_pos.torch,
            "hand joint_vel": h.joint_vel.torch,
            "hand body_pos_w": h.body_pos_w.torch,
            "hand body_vel_w": h.body_vel_w.torch,
            "object root_pos_w": o.root_pos_w.torch,
            "object root_quat_w": o.root_quat_w.torch,
            "object root_vel_w": o.root_vel_w.torch,
        }
        lines += ["", "-- raw physics state (cause B: dynamics divergence) --"]
        for name, t in tensors.items():
            lines.append(f"{name:18s}: {_finite(t)}")

        # Localize WHICH envs go bad (reduce over all non-env dims).
        def bad_envs(t: torch.Tensor):
            flat = t.reshape(self.num_envs, -1)
            mask = ~torch.isfinite(flat).all(dim=1)
            return mask.nonzero().flatten().tolist()

        lines += ["", "-- bad env localization --"]
        first_bad = None
        for name, t in tensors.items():
            be = bad_envs(t)
            if be:
                lines.append(f"{name:18s}: {len(be)} bad envs, first idx {be[:8]}")
                if first_bad is None:
                    first_bad = be[0]

        # Inspect the first diverging env's initial cube state vs the hand base
        # (cause-B sub-hypothesis: cube spawned interpenetrating / mis-placed).
        if first_bad is not None:
            i = first_bad
            base = h.body_pos_w.torch[i, 0]
            lines += [
                "",
                f"-- first bad env (idx {i}) state --",
                f"object pos (world): {o.root_pos_w.torch[i].tolist()}",
                f"object quat       : {o.root_quat_w.torch[i].tolist()}",
                f"object lin/ang vel: {o.root_vel_w.torch[i].tolist()}",
                f"hand base pos     : {base.tolist()}",
                f"|object - base|   : {float(torch.linalg.norm(o.root_pos_w.torch[i] - base)):.4g}",
            ]

        lines += ["", "-- derived obs term --", f"scale_transform(dof): {_finite(self._scaled_dof_probe())}"]
        with open(PROBE_OUT, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _scaled_dof_probe(self) -> torch.Tensor:
        from isaaclab.utils.math import scale_transform

        return scale_transform(
            self.hand.data.joint_pos.torch, self.hand_dof_lower_limits, self.hand_dof_upper_limits
        )

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        if not type(self)._probe_done:
            cls = type(self)
            n = cls._obs_calls
            cls._obs_calls += 1
            # call 0 == reset obs (cause A territory); call >=1 == after a physics step (cause B).
            when = "RESET-obs" if n == 0 else f"AFTER-STEP-{n}"
            policy = obs["policy"]
            bad = not bool(torch.isfinite(policy).all())
            if bad or n == 0:
                self._dump(when + (" NAN-DETECTED" if bad else " clean"))
            if bad:
                cls._probe_done = True
                raise RuntimeError(f"shadow NaN probe: non-finite obs at {when} -> see {PROBE_OUT}")
        return obs


@configclass
class ShadowHandReorientDebugCfg(ShadowHandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.sim.physics.newton_mjwarp.solver_cfg.njmax = 512
        self.sim.physics.newton_mjwarp.solver_cfg.nconmax = 150


def register():
    if TASK_ID in gym.registry:
        return
    gym.register(
        id=TASK_ID,
        entry_point="debug_env:DebugReorientEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "debug_env:ShadowHandReorientDebugCfg",
            "rsl_rl_cfg_entry_point": (
                "isaaclab_tasks.core.reorient.config.shadow_hand.agents.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg"
            ),
        },
    )
