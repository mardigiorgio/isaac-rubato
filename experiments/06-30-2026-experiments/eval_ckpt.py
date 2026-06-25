# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate an RSL-RL checkpoint for N steps and report mean EE tracking error (JSON).

Modeled on rsl_rl/play.py. Loads a checkpoint, runs the deterministic policy for --steps
control steps across --num_envs envs, and averages the ee_pose command metrics
(position_error [m], orientation_error [rad]) over the steps after --warmup. The solver
(fixed vs adaptive) is selected exactly as in training: NEWTON_ADAPTIVE=1 env var +
physics=newton_mjwarp preset. This lets us evaluate any checkpoint under EITHER solver.
"""

import argparse
import contextlib
import importlib.metadata as metadata
import json
import os
import sys

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import cli_args  # isort: skip

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint (tracking error).")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--steps", type=int, default=480, help="control steps to run")
parser.add_argument("--warmup", type=int, default=120, help="steps to skip before averaging")
parser.add_argument("--metrics_out", type=str, default=None, help="path to write JSON result")
parser.add_argument("--tag", type=str, default="", help="free label echoed into the JSON")
parser.add_argument("--external_callback", default=None)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        resume_path = retrieve_file_path(args_cli.checkpoint)
        env_cfg.log_dir = None

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        configure_seed(env_cfg.seed, True)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        cmd = env.unwrapped.command_manager.get_term("ee_pose")
        obs = env.get_observations()
        pos_sum = ori_sum = 0.0
        n = 0
        pos_hist = []
        for t in range(args_cli.steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
            if t >= args_cli.warmup:
                pe = cmd.metrics["position_error"].mean().item()
                oe = cmd.metrics["orientation_error"].mean().item()
                pos_sum += pe
                ori_sum += oe
                n += 1
                if t % 20 == 0:
                    pos_hist.append(round(pe, 5))

        result = {
            "tag": args_cli.tag,
            "checkpoint": resume_path,
            "task": args_cli.task,
            "num_envs": env_cfg.scene.num_envs,
            "steps": args_cli.steps,
            "warmup": args_cli.warmup,
            "adaptive_env": os.environ.get("NEWTON_ADAPTIVE", "0"),
            "mean_position_error_m": pos_sum / max(n, 1),
            "mean_orientation_error_rad": ori_sum / max(n, 1),
            "pos_err_samples": pos_hist,
        }
        print("EVAL_RESULT_JSON " + json.dumps(result))
        if args_cli.metrics_out:
            with open(args_cli.metrics_out, "w") as f:
                json.dump(result, f, indent=2)
        env.close()


if __name__ == "__main__":
    main()
