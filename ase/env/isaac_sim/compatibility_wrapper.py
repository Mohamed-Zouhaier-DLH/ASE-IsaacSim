"""
Compatibility wrapper to maintain RLGPUEnv interface for Isaac Sim environments.

This wrapper ensures that Isaac Sim-based environments can be used with the existing
RL training pipeline (rl_games) without requiring changes to the runner code.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


class IsaacSimVecEnvWrapper:
    """
    Wrapper that provides RLGPUEnv interface for Isaac Sim environments.
    
    This maintains compatibility with the existing rl_games training pipeline
    without requiring changes to the runner code.
    
    The wrapper exposes:
    - step(actions) -> obs, rewards, dones, extras
    - reset() -> obs
    - get_state() -> privileged_state
    - action_space, observation_space properties
    
    Usage:
        env = HumanoidAMPIsaacSim(cfg)
        wrapped_env = IsaacSimVecEnvWrapper(env)
        
        # Use with rl_games
        runner = Runner(wrapped_env)
        runner.run(cfg)
    """
    
    def __init__(self, isaac_sim_env):
        """
        Initialize the compatibility wrapper.
        
        Args:
            isaac_sim_env: Instance of BaseTaskIsaacSim or subclass
        """
        self.env = isaac_sim_env
        
        # Expose core properties
        self.num_envs = isaac_sim_env.num_envs
        self.num_obs = isaac_sim_env.num_obs
        self.num_actions = isaac_sim_env.num_actions
        self.num_states = isaac_sim_env.num_states
        self.device = isaac_sim_env.device
        self.max_episode_length = isaac_sim_env.max_episode_length
        
        # Direct access to buffers for efficiency
        self.obs_buf = isaac_sim_env.obs_buf
        self.reward_buf = isaac_sim_env.reward_buf
        self.reset_buf = isaac_sim_env.reset_buf
        self.state_buf = isaac_sim_env.state_buf
        self.progress_buf = isaac_sim_env.progress_buf
        
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step all environments with the given actions.
        
        Args:
            actions: Action tensor of shape (num_envs, num_actions)
            
        Returns:
            obs_buf: Observations (num_envs, num_obs)
            reward_buf: Rewards (num_envs,)
            reset_buf: Reset flags (num_envs,) - 1 = done, 0 = not done
            extras: Dictionary of additional information
        """
        obs, rewards, dones, extras = self.env.step(actions)
        return obs, rewards, dones, extras
    
    def reset(self) -> torch.Tensor:
        """
        Reset all environments.
        
        Returns:
            Observations for all environments (num_envs, num_obs)
        """
        return self.env.reset()
    
    def reset_idx(self, env_ids: torch.Tensor):
        """
        Reset specific environments.
        
        Args:
            env_ids: Tensor of environment indices to reset
        """
        self.env.reset_idx(env_ids)
    
    def get_state(self) -> torch.Tensor:
        """
        Get privileged state information for all environments.
        
        This is used for asymmetric training where the critic has access
        to privileged information not available to the actor.
        
        Returns:
            State tensor (num_envs, num_states)
        """
        return self.env.get_state()
    
    @property
    def action_space(self):
        """
        Return action space specification.
        
        Returns:
            Dictionary with 'shape' and 'dtype' keys
        """
        return self.env.action_space
    
    @property
    def observation_space(self):
        """
        Return observation space specification.
        
        Returns:
            Dictionary with 'shape' and 'dtype' keys
        """
        return self.env.observation_space
    
    def render(self, mode: str = 'human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human', 'rgb_array', etc.)
            
        Returns:
            Rendered image if mode='rgb_array', else None
        """
        # Isaac Sim handles rendering automatically during world.step()
        # This is here for interface compatibility
        pass
    
    def seed(self, seed: Optional[int] = None):
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed value
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def close(self):
        """Clean up environment resources."""
        if hasattr(self.env, 'cleanup'):
            self.env.cleanup()
    
    def __getattr__(self, name):
        """
        Forward any unknown attributes to the underlying environment.
        
        This allows direct access to environment-specific methods and properties.
        """
        return getattr(self.env, name)


class RLGPUEnvAdapter(IsaacSimVecEnvWrapper):
    """
    Explicit adapter that mimics the RLGPUEnv interface from Isaac Gym.
    
    This class can be used as a drop-in replacement for RLGPUEnv from
    rl_games.common.envs_utils when using Isaac Sim environments.
    
    Usage:
        # In run.py or training script:
        from ase.env.isaac_sim.compatibility_wrapper import RLGPUEnvAdapter
        from ase.env.isaac_sim.tasks.humanoid_amp import HumanoidAMPIsaacSim
        
        env = HumanoidAMPIsaacSim(cfg)
        rlgpu_env = RLGPUEnvAdapter(env)
        
        # Use with rl_games
        runner = Runner(rlgpu_env)
    """
    
    def __init__(self, isaac_sim_env):
        """
        Initialize the RLGPUEnv adapter.
        
        Args:
            isaac_sim_env: Instance of BaseTaskIsaacSim or subclass
        """
        super().__init__(isaac_sim_env)
        
        # RLGPUEnv specific attributes
        self.clip_obs = getattr(isaac_sim_env.cfg, 'clip_observations', float('inf'))
        self.clip_actions = getattr(isaac_sim_env.cfg, 'clip_actions', float('inf'))
        
    def get_number_of_agents(self) -> int:
        """
        Get the number of agents (environments).
        
        Returns:
            Number of parallel environments
        """
        return self.num_envs
    
    def get_env_info(self) -> Dict:
        """
        Get environment information for rl_games.
        
        Returns:
            Dictionary with environment specifications
        """
        return {
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'agents': self.num_envs,
            'value_size': 1,  # Standard for single-agent RL
        }


def create_rlgpu_env(task_name: str, cfg: Dict, **kwargs) -> RLGPUEnvAdapter:
    """
    Factory function to create RLGPUEnv-compatible environments from Isaac Sim tasks.
    
    This function provides a clean interface for creating environments that matches
    the pattern used in the original ASE codebase.
    
    Args:
        task_name: Name of the task (e.g., 'HumanoidAMP', 'HumanoidReach')
        cfg: Configuration dictionary
        **kwargs: Additional arguments passed to the task constructor
        
    Returns:
        RLGPUEnvAdapter wrapping the Isaac Sim task
        
    Example:
        env = create_rlgpu_env('HumanoidAMP', cfg, device='cuda:0', headless=True)
        runner = Runner(env)
    """
    # Import task classes
    from .tasks import TASK_REGISTRY
    
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {list(TASK_REGISTRY.keys())}"
        )
    
    # Create the task
    task_class = TASK_REGISTRY[task_name]
    task = task_class(cfg, **kwargs)
    
    # Wrap in adapter
    return RLGPUEnvAdapter(task)
