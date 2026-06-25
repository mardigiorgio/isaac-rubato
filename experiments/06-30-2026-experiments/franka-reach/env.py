"""Franka reach, repo-resident.

Stock franka-reach + the Newton constraint-buffer fix: franka-reach overflows the stock
njmax=50 (it needs >=53), which silently drops constraints. Lives in this experiment folder
so env + run script + scenes travel together. Loaded by train.sh via `--external_callback
env.register`.
"""

import os

import gymnasium as gym
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.core.reach.config.franka.joint_pos_env_cfg import FrankaReachEnvCfg

TASK_ID = "Isaac-Reach-Franka-Rubato"


@configclass
class FrankaReachRubatoCfg(FrankaReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # franka-reach overflows the stock Newton buffer; give it headroom
        self.sim.physics.newton_mjwarp.solver_cfg.njmax = 128
        self.sim.physics.newton_mjwarp.solver_cfg.nconmax = 32
        # Diagnostic toggle (off by default): disable the action_rate/joint_vel penalty curriculum
        # (the 50x/10x weight ramp at ~iter 188). Tests whether that effort-penalty ramp is what
        # drives the adaptive-solver position-tracking decline. Set NO_EFFORT_CURRICULUM=1 to keep
        # the effort penalties at their tiny base weight for the whole run.
        if os.environ.get("NO_EFFORT_CURRICULUM") == "1":
            self.curriculum.action_rate = None
            self.curriculum.joint_vel = None


def register():
    if TASK_ID in gym.registry:
        return
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "env:FrankaReachRubatoCfg",
            "rsl_rl_cfg_entry_point": (
                "isaaclab_tasks.core.reach.config.franka.agents.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg"
            ),
        },
    )
