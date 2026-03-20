# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an environment instance to EvoTorch environment.

The following example shows how to wrap an environment for EvoTorch:

.. code-block:: python

    from isaaclab_ea.evotorch import EvoTorchVecEnvWrapper

    env = EvoTorchVecEnvWrapper(env)

"""

from __future__ import annotations
# needed to import for type hinting: Agent | list[Agent]
import torch

from typing import Literal
from torch.func import functional_call, vmap

from isaaclab.envs import ManagerBasedRLEnv
from evotorch.neuroevolution import NEProblem
from evotorch import Problem
from evotorch.algorithms import CEM, CMAES, Cosyne, GeneticAlgorithm, MAPElites, PGPE, SNES, XNES
from evotorch.logging import PicklingLogger, StdOutLogger

# TODO: create Runner

class Runner:
    def __init__(self, env_problem, algorithm_name: str, algorithm_cfg: dict, logger_cfg:dict):
        algorithm_cfg["problem"] = env_problem
        algorithm_name = algorithm_name.upper()
        if algorithm_name != "MAPELITES":
            algorithm_cfg["popsize"] = env_problem.num_envs
        
        if algorithm_name == "CEM":
            pass
        elif algorithm_name == "CMAES":
            pass
        elif algorithm_name == "COSYNE":
            pass
        elif algorithm_name == "GA":
            pass
        elif algorithm_name == "MAPELITES":
            pass
        elif algorithm_name == "PGPE":            
            self.searcher = PGPE(**algorithm_cfg)
        elif algorithm_name == "SNES":
            pass
        elif algorithm_name == "XNES":
            pass
        
        self.logger = PicklingLogger(self.searcher, **logger_cfg)
        self.stdout_logger = StdOutLogger(self.searcher)
        
    
    def run(self, n_runs):
        self.searcher.run(n_runs)
    
"""
Vectorized environment wrapper.
"""
    
class EvoTorchNEVecEnvWrapper(NEProblem):
    def __init__(self, env: ManagerBasedRLEnv, problem_cfg): # TODO: add config to init
        # initialize the wrapper
        self.env = env
        
        # store information required by wrapper
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_envs = self.unwrapped.num_envs
        self.render_mode = self.unwrapped.render_mode
        sim_device = self.unwrapped.device
        
        # Get observation dimension (handle tuple or int)
        obs_dim_raw = self.unwrapped.observation_manager.group_obs_dim["policy"]
        self.obs_dim = obs_dim_raw[0] if isinstance(obs_dim_raw, tuple) else int(obs_dim_raw)

        # Get action dimension (handle tuple or int)
        act_dim_raw = self.unwrapped.action_manager.total_action_dim
        self.act_dim = act_dim_raw[0] if isinstance(act_dim_raw, tuple) else int(act_dim_raw)

        network_expression = problem_cfg.pop("network_expression")
        network_cfg = network_expression.format(
            obs_dim = self.obs_dim,
            act_dim = self.act_dim
        )
        
        problem_cfg["network"] = network_cfg
        problem_cfg["device"] = sim_device
        
        super().__init__(**problem_cfg)
        
        # Create a "dummy" genome filled with zeros of the exact size calculated by NEProblem
        dummy_weights = torch.zeros(self.solution_length, device=self.device)
        
        # Ask NEProblem to build a real instance of the network
        self.template_net = self.parameterize_net(dummy_weights)

        # Extract the properties using the template network
        self.param_names = [name for name, _ in self.template_net.named_parameters()]
        self.param_shapes = [p.shape for p in self.template_net.parameters()]
        self.param_numels = [p.numel() for p in self.template_net.parameters()]
        
    """
    Properties
    """
    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    def _evaluate_batch(self, batch):
        """
        Called by EvoTorch to evaluate a population of solutions.
        'batch' contains the candidate weight vectors.
        """
        # Extract population weights: [num_envs, num_weights]
        population_weights = batch.values 
        
        # Reset environment and get initial observations
        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"]
        
        # Store accumulated rewards for each individual
        fitness = torch.zeros(self.num_envs, device=self.device)
        
        # Track which individuals are still active
        alive_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Simulation loop
        for step in range(self.max_episode_length):
            
            # Compute actions for the whole population in one pass
            actions = self._batched_forward_pass(obs, population_weights)
            
            # Step environment
            obs_dict, rewards, terminated, truncated, _ = self.env.step(actions)
            obs = obs_dict["policy"]
            
            # Add rewards only for active individuals
            fitness += rewards * alive_mask
            
            # Update done flags
            dones = terminated | truncated
            alive_mask = alive_mask & ~dones
            
            # Stop early if all individuals are done
            if not alive_mask.any():
                break
            
        # Send fitness values back to EvoTorch
        batch.set_evals(fitness)
        
    def _batched_forward_pass(self, obs, population_weights):
        # Reconstruct the parameter dictionary from the flat weight vector
        params_dict = {}
        start_idx = 0
        for name, shape, numel in zip(self.param_names, self.param_shapes, self.param_numels):
            param_slice = population_weights[:, start_idx : start_idx + numel]
            params_dict[name] = param_slice.view(-1, *shape)
            start_idx += numel
            
        # PyTorch vmap: Apply all genomes to all observations in a single pass
        def compute_action(params, single_obs):
            return functional_call(self.template_net, params, single_obs)

        batched_compute = vmap(compute_action, in_dims=(0, 0))
        
        return batched_compute(params_dict, obs)