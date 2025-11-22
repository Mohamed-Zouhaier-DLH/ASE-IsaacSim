"""
Base environment class for Isaac Sim implementation of ASE tasks.

This provides the foundation for all ASE tasks in Isaac Sim, handling:
- World creation and management
- Multi-instance environment setup
- Common observation/action buffer management
- Physics stepping
- Reset logic
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class BaseTaskIsaacSim:
    """
    Base class for ASE tasks in Isaac Sim.
    
    This class provides the core functionality needed to run ASE tasks in Isaac Sim,
    including multi-instance management, state tracking, and the step/reset interface.
    
    Subclasses should implement:
    - _create_envs(): Set up articulations and scene
    - _compute_observations(): Generate observation tensors
    - _compute_reward(): Calculate rewards
    - _compute_reset(): Determine which environments should reset
    - pre_physics_step(actions): Apply actions before physics
    - post_physics_step(): Update state after physics
    """
    
    def __init__(
        self, 
        cfg: Dict, 
        device: str = 'cuda:0',
        headless: bool = True
    ):
        """
        Initialize the base Isaac Sim task.
        
        Args:
            cfg: Task configuration dictionary
            device: PyTorch device for tensors
            headless: Whether to run without rendering
        """
        self.cfg = cfg
        self.device = device
        self.headless = headless
        
        # Environment configuration
        self.num_envs = cfg.get('num_envs', 1024)
        self.num_obs = cfg.get('num_obs', 0)  # Set by subclass
        self.num_actions = cfg.get('num_actions', 0)  # Set by subclass
        self.num_states = cfg.get('num_states', 0)  # Privileged state size
        
        # Physics parameters
        self.dt = cfg.get('dt', 1.0 / 60.0)
        self.substeps = cfg.get('substeps', 2)
        self.control_freq_inv = cfg.get('control_freq_inv', 1)
        
        # Buffers for observations, rewards, resets
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
        self.reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.state_buf = torch.zeros((self.num_envs, self.num_states), device=self.device, dtype=torch.float32)
        
        # Extras dictionary for additional info
        self.extras = {}
        
        # Episode length tracking
        self.max_episode_length = cfg.get('max_episode_length', 1000)
        
        # Isaac Sim world (to be initialized by subclass)
        self.world = None
        self.articulations = []
        
        # Initialize the environment
        self._create_sim()
        self._create_envs()
        
    def _create_sim(self):
        """
        Create the Isaac Sim world.
        
        This method initializes the Omniverse simulation environment.
        Subclasses can override to customize world creation.
        """
        try:
            from omni.isaac.kit import SimulationApp
            from omni.isaac.core import World
            
            # Create simulation app if not already created
            if not hasattr(self, 'simulation_app'):
                simulation_config = {
                    "headless": self.headless,
                }
                self.simulation_app = SimulationApp(simulation_config)
            
            # Create world
            self.world = World(
                stage_units_in_meters=1.0,
                physics_dt=self.dt,
                rendering_dt=self.dt,
                backend='torch',
                device=self.device
            )
            
            # Set physics scene parameters
            from omni.isaac.core.utils.physics import set_physics_scene_parameters
            set_physics_scene_parameters(
                physics_dt=self.dt / self.substeps,
                num_position_iterations=self.cfg.get('num_position_iterations', 4),
                num_velocity_iterations=self.cfg.get('num_velocity_iterations', 0),
                contact_offset=self.cfg.get('contact_offset', 0.02),
                rest_offset=self.cfg.get('rest_offset', 0.0),
            )
            
        except ImportError as e:
            raise ImportError(
                "Isaac Sim is not installed or not properly configured. "
                "Please install Isaac Sim from NVIDIA Omniverse."
            ) from e
    
    def _create_envs(self):
        """
        Create environment instances (articulations, terrain, etc.).
        
        This method should be implemented by subclasses to:
        1. Load USD assets
        2. Create articulation instances
        3. Set up terrain/obstacles if needed
        4. Initialize any task-specific objects
        """
        raise NotImplementedError("Subclasses must implement _create_envs()")
    
    def reset(self) -> torch.Tensor:
        """
        Reset all environments.
        
        Returns:
            Observations for all environments
        """
        reset_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_idx(reset_env_ids)
        return self.obs_buf
    
    def reset_idx(self, env_ids: torch.Tensor):
        """
        Reset specific environments.
        
        Args:
            env_ids: Indices of environments to reset
        """
        # Reset progress tracking
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        
        # Call subclass-specific reset logic
        self._reset_idx(env_ids)
        
        # Compute observations for reset environments
        self._compute_observations()
    
    def _reset_idx(self, env_ids: torch.Tensor):
        """
        Subclass-specific reset logic.
        
        Args:
            env_ids: Indices of environments to reset
        """
        raise NotImplementedError("Subclasses must implement _reset_idx()")
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step the environments with the given actions.
        
        Args:
            actions: Action tensor of shape (num_envs, num_actions)
            
        Returns:
            observations: Observation tensor (num_envs, num_obs)
            rewards: Reward tensor (num_envs,)
            dones: Done flags (num_envs,)
            extras: Dictionary of additional information
        """
        # Apply actions
        self.pre_physics_step(actions)
        
        # Step physics
        for _ in range(self.control_freq_inv):
            self.world.step(render=not self.headless)
        
        # Post-physics updates
        self.post_physics_step()
        
        # Compute observations, rewards, and resets
        self._compute_observations()
        self._compute_reward()
        self._compute_reset()
        
        # Update progress
        self.progress_buf += 1
        
        # Auto-reset based on episode length
        timeout_reset = self.progress_buf >= self.max_episode_length
        self.reset_buf = torch.where(timeout_reset, torch.ones_like(self.reset_buf), self.reset_buf)
        
        # Reset environments that are done
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        
        return self.obs_buf, self.reward_buf, self.reset_buf, self.extras
    
    def pre_physics_step(self, actions: torch.Tensor):
        """
        Apply actions before physics step.
        
        Args:
            actions: Action tensor to apply
        """
        raise NotImplementedError("Subclasses must implement pre_physics_step()")
    
    def post_physics_step(self):
        """
        Update state after physics step.
        
        This is called after the physics simulation has stepped.
        Subclasses should use this to update internal state, buffers, etc.
        """
        raise NotImplementedError("Subclasses must implement post_physics_step()")
    
    def _compute_observations(self):
        """
        Compute observations for all environments.
        
        This should populate self.obs_buf with the latest observations.
        """
        raise NotImplementedError("Subclasses must implement _compute_observations()")
    
    def _compute_reward(self):
        """
        Compute rewards for all environments.
        
        This should populate self.reward_buf with the latest rewards.
        """
        raise NotImplementedError("Subclasses must implement _compute_reward()")
    
    def _compute_reset(self):
        """
        Determine which environments should reset.
        
        This should populate self.reset_buf with reset flags (1 = reset, 0 = continue).
        """
        raise NotImplementedError("Subclasses must implement _compute_reset()")
    
    def get_state(self) -> torch.Tensor:
        """
        Get privileged state information.
        
        Returns:
            State tensor (num_envs, num_states)
        """
        return self.state_buf
    
    @property
    def action_space(self):
        """Return action space specification."""
        return {
            'shape': (self.num_actions,),
            'dtype': np.float32
        }
    
    @property
    def observation_space(self):
        """Return observation space specification."""
        return {
            'shape': (self.num_obs,),
            'dtype': np.float32
        }
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'simulation_app') and self.simulation_app is not None:
            self.simulation_app.close()


class MultiArticulationMixin:
    """
    Mixin class providing utilities for managing multiple articulations.
    
    This can be used by task classes to simplify batch operations on articulations.
    """
    
    def create_articulations(self, usd_path: str, num_instances: int, spacing: float = 2.0):
        """
        Create multiple instances of an articulation.
        
        Args:
            usd_path: Path to the USD asset
            num_instances: Number of instances to create
            spacing: Spacing between instances
            
        Returns:
            List of Articulation objects
        """
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.stage import add_reference_to_stage
        import omni.isaac.core.utils.prims as prim_utils
        
        articulations = []
        rows = int(np.sqrt(num_instances))
        cols = int(np.ceil(num_instances / rows))
        
        for i in range(num_instances):
            row = i // cols
            col = i % cols
            
            # Position in grid
            x = col * spacing - (cols - 1) * spacing / 2
            y = row * spacing - (rows - 1) * spacing / 2
            
            prim_path = f"/World/Env_{i}/Articulation"
            
            # Add reference to stage
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            
            # Create articulation object
            articulation = Articulation(prim_path=prim_path)
            articulation.set_world_pose(position=np.array([x, y, 1.0]))
            
            # Add to scene
            self.world.scene.add(articulation)
            articulations.append(articulation)
        
        return articulations
    
    def get_batch_joint_positions(self, articulations: List) -> torch.Tensor:
        """
        Get joint positions for all articulations.
        
        Args:
            articulations: List of Articulation objects
            
        Returns:
            Joint positions tensor (num_articulations, num_joints)
        """
        positions = []
        for art in articulations:
            pos = art.get_joint_positions()
            positions.append(pos)
        return torch.stack(positions)
    
    def get_batch_joint_velocities(self, articulations: List) -> torch.Tensor:
        """
        Get joint velocities for all articulations.
        
        Args:
            articulations: List of Articulation objects
            
        Returns:
            Joint velocities tensor (num_articulations, num_joints)
        """
        velocities = []
        for art in articulations:
            vel = art.get_joint_velocities()
            velocities.append(vel)
        return torch.stack(velocities)
    
    def set_batch_joint_position_targets(self, articulations: List, targets: torch.Tensor):
        """
        Set joint position targets for all articulations.
        
        Args:
            articulations: List of Articulation objects
            targets: Target positions tensor (num_articulations, num_joints)
        """
        for i, art in enumerate(articulations):
            art.set_joint_position_targets(targets[i])
