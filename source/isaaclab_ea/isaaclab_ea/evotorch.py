# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an environment instance to EvoTorch environment.

The following example shows how to wrap an environment for EvoTorch:

.. code-block:: python

    from isaaclab_ea.evotorch import EvoTorchVecEnvWrapper

    env = EvoTorchVecEnvWrapper(env)  # or ml_framework="jax"

"""

from __future__ import annotations
# needed to import for type hinting: Agent | list[Agent]
import torch

from typing import Literal

from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedRLEnv
from evotorch import Problem

"""
Vectorized environment wrapper.
"""

class EvoTorchVecEnvWrapper(Problem):
    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        
        # Guardamos información usando self.unwrapped
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_envs = self.unwrapped.num_envs
        self.render_mode = self.unwrapped.render_mode
        
        # IMPORTANTE: Usar self.unwrapped para acceder a los managers
        obs_dim_raw = self.unwrapped.observation_manager.group_obs_dim["policy"]
        self.obs_dim = obs_dim_raw[0] if isinstance(obs_dim_raw, tuple) else int(obs_dim_raw)

        # En ActionManager usamos total_action_dim, que devuelve un entero directamente
        act_dim_raw = self.unwrapped.action_manager.total_action_dim
        self.act_dim = act_dim_raw[0] if isinstance(act_dim_raw, tuple) else int(act_dim_raw)

        self.num_weights = (self.obs_dim * self.act_dim) + self.act_dim
        
        sim_device = self.unwrapped.device  # Cambiado a self.device para consistencia
        # Inicializar la clase Problem de EvoTorch
        super().__init__(
            objective_sense="max",  
            solution_length=self.num_weights,
            initial_bounds=(-1.0, 1.0),
            device=sim_device, # Ahora sí encuentra self.device
        )
        
    """
    Properties
    """
    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["policy"], {"observations": obs_dict}

    def _evaluate_batch(self, batch):
        """
        Este método es llamado por EvoTorch. 
        'batch' contiene los pesos de la población actual.
        """
        # 1. Extraer la población (Matriz de tamaño [num_envs, num_weights])
        population_weights = batch.values 
        
        # 2. Resetear el entorno de Isaac Lab
        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"]
        
        # Tensor para acumular el fitness (recompensa) de cada individuo
        fitness = torch.zeros(self.num_envs, device=self.device)
        
        # MÁSCARA: Todos los individuos empiezan vivos (True)
        alive_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 3. Bucle de Simulación
        for step in range(self.max_episode_length):
            
            # --- LA MAGIA: Inferencia Vectorizada ---
            # Pasamos las observaciones y la población de pesos para obtener las acciones
            actions = self._batched_forward_pass(obs, population_weights)
            
            # Aplicar acciones en Isaac Lab
            obs_dict, rewards, terminated, truncated, _ = self.env.step(actions)
            obs = obs_dict["policy"]
            
            # Acumular recompensas
            fitness += rewards * alive_mask
            
            dones = terminated | truncated
            alive_mask = alive_mask & ~dones
            
            if not alive_mask.any():
                break
            
        # 4. Devolver el fitness a EvoTorch
        batch.set_evals(fitness)
        
    def _batched_forward_pass(self, obs, weights):
        """
        Convierte el vector plano de pesos en matrices y hace la inferencia.
        Ejemplo para una red de 1 sola capa sin capas ocultas (Linear).
        """
        # Separar pesos (W) y sesgos (b)
        W_size = self.obs_dim * self.act_dim
        
        # W: [num_envs, obs_dim, act_dim]
        W = weights[:, :W_size].view(self.num_envs, self.obs_dim, self.act_dim)
        # b: [num_envs, act_dim]
        b = weights[:, W_size:] 
        
        # Multiplicación de matrices batcheada (bmm)
        # obs es [num_envs, obs_dim]. Lo expandimos a [num_envs, 1, obs_dim]
        obs_exp = obs.unsqueeze(1) 
        
        # [num_envs, 1, obs_dim] x [num_envs, obs_dim, act_dim] = [num_envs, 1, act_dim]
        actions = torch.bmm(obs_exp, W).squeeze(1) + b
        
        # Devolver acciones (generalmente acotadas entre -1 y 1)
        return torch.tanh(actions)