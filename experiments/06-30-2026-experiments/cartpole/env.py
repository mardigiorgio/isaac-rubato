"""Cartpole, repo-resident.

Stock Isaac Lab cartpole, registered here as this experiment's env (and edit-home). No
overrides yet -- cartpole runs stock; this file is where any tweak would go. Loaded by
train.sh via `--external_callback env.register`.
"""

import gymnasium as gym
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleEnvCfg

TASK_ID = "Isaac-Cartpole-Rubato"


@configclass
class CartpoleRubatoCfg(CartpoleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # no overrides yet -- stock cartpole. Add study-specific tweaks here.


def register():
    if TASK_ID in gym.registry:
        return
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "env:CartpoleRubatoCfg",
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        },
    )
