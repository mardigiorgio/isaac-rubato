"""Env for this experiment.

Subclass a stock Isaac Lab task, add your edits in __post_init__, register a gym id.
Lives in the experiment folder so the env + run script + scenes travel together. train.sh
loads it via `--external_callback env.register` (this folder is on PYTHONPATH, so the module
is importable as `env`).

To author a custom scene: build it in the GUI (`./isaac-rubato`), Save As into ./scenes/,
then reference it here, e.g.
    import os, isaaclab.sim as sim_utils
    SCENES = os.path.join(os.path.dirname(__file__), "scenes")
    self.scene.<asset>.spawn = sim_utils.UsdFileCfg(usd_path=os.path.join(SCENES, "my.usd"))
"""

import gymnasium as gym
from isaaclab.utils.configclass import configclass

# from isaaclab_tasks.core.<task>.<...>_env_cfg import StockEnvCfg

TASK_ID = "Isaac-Example-Rubato"


# @configclass
# class ExampleRubatoCfg(StockEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         # your edits here (rewards, randomization, Newton buffers, custom scene asset, ...)


def register():
    if TASK_ID in gym.registry:
        return
    raise NotImplementedError("Fill in env.py: import the stock cfg, subclass it, gym.register below.")
    # gym.register(
    #     id=TASK_ID,
    #     entry_point="isaaclab.envs:ManagerBasedRLEnv",
    #     disable_env_checker=True,
    #     kwargs={
    #         "env_cfg_entry_point": "env:ExampleRubatoCfg",
    #         "rsl_rl_cfg_entry_point": "isaaclab_tasks.core.<task>...:<PPORunnerCfg>",
    #     },
    # )
