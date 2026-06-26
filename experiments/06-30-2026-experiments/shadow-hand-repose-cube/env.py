"""Shadow Hand in-hand cube reorient, repo-resident.

Stock Isaac Lab `Isaac-Reorient-Cube-Shadow-Direct` (a DirectRLEnv), re-registered here as
this experiment's env + edit-home so env + run script + scenes travel together. Loaded via the
standard isaaclab.sh train command with `--external_callback env.register` (registers the task +
sets the constraint-buffer sizes here).

This is the contact-rich task the adaptive-solver study targets: stiff fingertip/cube contact
is exactly where step-doubling subdivides time. The solver variant is selected by the
`--solver {mujoco,mujoco-adaptive,sap,sap-adaptive}` CLI flag, not here -- this file only carries
task-level edits.
"""

import gymnasium as gym
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg

TASK_ID = "Isaac-Reorient-Cube-Shadow-Rubato"


@configclass
class ShadowHandReorientCube(ShadowHandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Stock njmax=200 / nconmax=70 OVERFLOW on cube-in-hand contact: MuJoCo-Warp
        # reports "nefc overflow - please increase njmax" peaking ~377, which drops
        # constraints and sends observations to NaN (rsl_rl check_nan aborts). Size the
        # constraint buffers above the observed peak with headroom. Cost: per-world
        # constraint arrays scale with njmax x num_envs -- drop NUM_ENVS if memory-bound.
        self.sim.physics.newton_mjwarp.solver_cfg.njmax = 512
        self.sim.physics.newton_mjwarp.solver_cfg.nconmax = 150
        # Adaptive error tolerance for the contact-rich regime (ignored when fixed).
        self.sim.physics.newton_mjwarp.solver_cfg.adaptive_tol = 1e-3
        # your edits here (rewards, randomization, custom scene asset, ...)


def register():
    if TASK_ID in gym.registry:
        return
    gym.register(
        id=TASK_ID,
        # Reorient is a DirectRLEnv (NOT ManagerBasedRLEnv) -- use the stock direct env class.
        entry_point="isaaclab_tasks.core.reorient.reorient_direct_env:ReorientDirectEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "env:ShadowHandReorientCube",
            "rsl_rl_cfg_entry_point": (
                "isaaclab_tasks.core.reorient.config.shadow_hand.agents.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg"
            ),
        },
    )
