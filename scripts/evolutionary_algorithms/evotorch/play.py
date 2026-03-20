# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a trained RL/EA agent.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained EA agent from EvoTorch.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate (default: 32 for visualization).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the pickled model checkpoint (.pkl).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import pickle

from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)

from isaaclab_ea.evotorch import EvoTorchNEVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
agent_cfg_entry_point = "evotorch_cfg_entry_point"

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with trained agent."""
    
    # env config
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 32
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create Isaaclab env
    print(f"[INFO] Creating environment: {args_cli.task}...")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # load evotorch checkpoint
    print(f"[INFO] Loading checkpoint from: {args_cli.checkpoint}")
    with open(args_cli.checkpoint, "rb") as f:
        checkpoint = pickle.load(f)
    
    # extract the best_solution from the pkl saved by the PickleLogger
    if "center" in checkpoint:
        best_solution = checkpoint["center"]
        print("[INFO] Loaded 'center' solution (typical for PGPE/CMAES).")
    elif "best" in checkpoint:
        best_solution = checkpoint["best"]
        print("[INFO] Loaded 'best' solution (typical for GA).")
    elif "pop_best" in checkpoint:
        best_solution = checkpoint["pop_best"]
        print("[INFO] Loaded 'pop_best' solution.")
    else:
        raise KeyError("Could not find 'center', 'best', or 'pop_best' in the checkpoint.")

    # extract the best solution's weights tensor 
    if isinstance(best_solution, torch.Tensor):
        best_weights = best_solution.to(env.unwrapped.device)
    else:
        best_weights = best_solution.values.to(env.unwrapped.device)

    # we use the wrapper to generate the net using the weights
    print("[INFO] Reconstructing neural network architecture...")
    problem = EvoTorchNEVecEnvWrapper(env, agent_cfg[agent_cfg["problem"]])
    
    policy_net = problem.parameterize_net(best_weights).to(env.unwrapped.device)
    
    policy_net.eval()

    # simulation loop
    print("[INFO] Starting play loop. Press Ctrl+C to exit.")
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy_net(obs)
            
        obs_dict, _, _, _, _ = env.step(actions)
        obs = obs_dict["policy"]

    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()