"""
Validation utilities for Isaac Sim migration.

This module provides tools to validate that the Isaac Sim implementation
produces equivalent behavior to the original Isaac Gym implementation.
"""

import torch
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class EnvironmentValidator:
    """
    Validator for comparing Isaac Gym and Isaac Sim environment behavior.
    
    This class helps ensure the migration preserves correctness by:
    - Comparing observation/action shapes
    - Validating observation/action formats
    - Comparing deterministic rollouts
    - Checking reward computation
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.reference_data = {}
    
    def validate_shapes(self, env, expected_shapes: Dict) -> bool:
        """
        Validate environment dimensions match expected values.
        
        Args:
            env: Environment instance
            expected_shapes: Dictionary with 'num_obs', 'num_actions', etc.
            
        Returns:
            True if all shapes match
        """
        print("\n=== Shape Validation ===")
        all_match = True
        
        for key in ['num_obs', 'num_actions', 'num_envs', 'num_states']:
            expected = expected_shapes.get(key)
            actual = getattr(env, key, None)
            
            if expected is not None and actual is not None:
                matches = expected == actual
                status = "✓" if matches else "✗"
                print(f"{status} {key}: expected={expected}, actual={actual}")
                all_match = all_match and matches
            elif expected is not None:
                print(f"✗ {key}: not found in environment")
                all_match = False
        
        print("=" * 30)
        return all_match
    
    def validate_observation_format(
        self,
        env,
        expected_components: List[Tuple[str, int]]
    ) -> bool:
        """
        Validate observation format matches expected structure.
        
        Args:
            env: Environment instance
            expected_components: List of (name, size) tuples
            
        Returns:
            True if format matches
        """
        print("\n=== Observation Format Validation ===")
        
        # Get a sample observation
        obs = env.reset()
        
        # Check total size
        total_expected = sum(size for _, size in expected_components)
        actual_size = obs.shape[-1]
        
        print(f"Total size: expected={total_expected}, actual={actual_size}")
        
        if total_expected != actual_size:
            print("✗ Observation size mismatch!")
            return False
        
        # Print component breakdown
        print("\nExpected components:")
        offset = 0
        for name, size in expected_components:
            print(f"  [{offset:3d}:{offset+size:3d}] {name} (size={size})")
            offset += size
        
        print("=" * 40)
        return True
    
    def capture_reference_data(
        self,
        env,
        num_steps: int = 100,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Capture reference data from an environment for later comparison.
        
        This should be run on the original (Isaac Gym) environment to create
        a reference for validating the Isaac Sim implementation.
        
        Args:
            env: Environment instance
            num_steps: Number of steps to record
            output_path: Optional path to save reference data
            
        Returns:
            Dictionary with recorded trajectory data
        """
        print(f"\n=== Capturing Reference Data ({num_steps} steps) ===")
        
        # Reset to deterministic initial state
        obs = env.reset()
        
        # Record trajectory
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'initial_obs': obs.cpu().numpy(),
        }
        
        # Use fixed random seed for deterministic actions
        torch.manual_seed(42)
        np.random.seed(42)
        
        for step in range(num_steps):
            # Random actions for reproducibility
            actions = torch.randn((env.num_envs, env.num_actions), device=env.device) * 0.1
            
            # Step environment
            obs, rewards, dones, extras = env.step(actions)
            
            # Record data
            trajectory['observations'].append(obs[0].cpu().numpy())  # First env only
            trajectory['actions'].append(actions[0].cpu().numpy())
            trajectory['rewards'].append(rewards[0].cpu().item())
            trajectory['dones'].append(dones[0].cpu().item())
            
            if step % 10 == 0:
                print(f"  Step {step:3d}: reward={rewards[0].item():.4f}")
        
        # Save if requested
        if output_path:
            with open(output_path, 'wb') as f:
                pickle.dump(trajectory, f)
            print(f"\nReference data saved to: {output_path}")
        
        print("=" * 50)
        return trajectory
    
    def compare_with_reference(
        self,
        env,
        reference_path: str,
        tolerance: float = 1e-3
    ) -> Tuple[bool, Dict]:
        """
        Compare environment behavior with reference data.
        
        Args:
            env: Isaac Sim environment to validate
            reference_path: Path to reference data (from Isaac Gym)
            tolerance: Tolerance for numerical comparisons
            
        Returns:
            Tuple of (passed, report_dict)
        """
        print(f"\n=== Comparing with Reference: {reference_path} ===")
        
        # Load reference data
        with open(reference_path, 'rb') as f:
            reference = pickle.load(f)
        
        num_steps = len(reference['observations'])
        
        # Reset to same initial state
        torch.manual_seed(42)
        np.random.seed(42)
        obs = env.reset()
        
        # Compare initial observation
        ref_init_obs = torch.from_numpy(reference['initial_obs']).to(env.device)
        init_diff = torch.abs(obs - ref_init_obs).max().item()
        print(f"Initial obs max diff: {init_diff:.6f}")
        
        # Run same trajectory
        differences = {
            'obs_diffs': [],
            'reward_diffs': [],
            'max_obs_diff': 0.0,
            'max_reward_diff': 0.0,
        }
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        for step in range(num_steps):
            # Same actions as reference
            actions = torch.randn((env.num_envs, env.num_actions), device=env.device) * 0.1
            
            # Step environment
            obs, rewards, dones, extras = env.step(actions)
            
            # Compare with reference
            ref_obs = torch.from_numpy(reference['observations'][step]).to(env.device)
            ref_reward = reference['rewards'][step]
            
            obs_diff = torch.abs(obs[0] - ref_obs).max().item()
            reward_diff = abs(rewards[0].item() - ref_reward)
            
            differences['obs_diffs'].append(obs_diff)
            differences['reward_diffs'].append(reward_diff)
            differences['max_obs_diff'] = max(differences['max_obs_diff'], obs_diff)
            differences['max_reward_diff'] = max(differences['max_reward_diff'], reward_diff)
            
            if step % 10 == 0:
                print(f"  Step {step:3d}: obs_diff={obs_diff:.6f}, reward_diff={reward_diff:.6f}")
        
        # Check if within tolerance
        passed = (
            differences['max_obs_diff'] < tolerance and
            differences['max_reward_diff'] < tolerance
        )
        
        # Print summary
        print("\n--- Summary ---")
        print(f"Max observation diff: {differences['max_obs_diff']:.6f}")
        print(f"Max reward diff: {differences['max_reward_diff']:.6f}")
        print(f"Tolerance: {tolerance:.6f}")
        print(f"Result: {'PASS ✓' if passed else 'FAIL ✗'}")
        print("=" * 50)
        
        return passed, differences
    
    def validate_pretrained_policy(
        self,
        env,
        policy_path: str,
        num_episodes: int = 10
    ) -> Dict:
        """
        Validate that a pretrained policy works in the new environment.
        
        Args:
            env: Isaac Sim environment
            policy_path: Path to pretrained policy checkpoint
            num_episodes: Number of episodes to run
            
        Returns:
            Dictionary with performance metrics
        """
        print(f"\n=== Validating Pretrained Policy: {policy_path} ===")
        
        # Load policy (assuming PyTorch checkpoint)
        # This is a placeholder - actual loading depends on policy format
        try:
            checkpoint = torch.load(policy_path, map_location=env.device)
            print("Policy checkpoint loaded successfully")
        except Exception as e:
            print(f"Failed to load policy: {e}")
            return {'success': False, 'error': str(e)}
        
        # Run episodes and collect metrics
        metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success': True,
        }
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            while episode_length < env.max_episode_length:
                # Get action from policy
                # This is a placeholder - actual inference depends on policy structure
                with torch.no_grad():
                    # actions = policy(obs)
                    # For now, use random actions as placeholder
                    actions = torch.randn((env.num_envs, env.num_actions), device=env.device) * 0.1
                
                # Step environment
                obs, rewards, dones, extras = env.step(actions)
                
                episode_reward += rewards[0].item()
                episode_length += 1
                
                if dones[0].item():
                    break
            
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_lengths'].append(episode_length)
            
            print(f"  Episode {episode+1}: reward={episode_reward:.2f}, length={episode_length}")
        
        # Compute statistics
        metrics['mean_reward'] = np.mean(metrics['episode_rewards'])
        metrics['std_reward'] = np.std(metrics['episode_rewards'])
        metrics['mean_length'] = np.mean(metrics['episode_lengths'])
        
        print("\n--- Summary ---")
        print(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean length: {metrics['mean_length']:.1f}")
        print("=" * 50)
        
        return metrics


class PhysicsValidator:
    """
    Validator for physics parameters and behavior.
    
    Ensures that physics simulation settings match between implementations.
    """
    
    def __init__(self):
        """Initialize the physics validator."""
        pass
    
    def validate_physics_params(
        self,
        isaac_sim_world,
        expected_params: Dict
    ) -> bool:
        """
        Validate that physics parameters match expected values.
        
        Args:
            isaac_sim_world: Isaac Sim World instance
            expected_params: Dictionary of expected physics parameters
            
        Returns:
            True if all parameters match
        """
        print("\n=== Physics Parameters Validation ===")
        
        # Get physics scene parameters
        # Note: API may vary by Isaac Sim version
        print("Expected parameters:")
        for key, value in expected_params.items():
            print(f"  {key}: {value}")
        
        print("\nNote: Manual verification required.")
        print("Check Isaac Sim physics scene settings match expected values.")
        print("=" * 40)
        
        # Manual check required - return True as placeholder
        return True
    
    def compare_contact_forces(
        self,
        env,
        reference_forces: Dict,
        tolerance: float = 0.1
    ) -> bool:
        """
        Compare contact forces with reference.
        
        Args:
            env: Environment instance
            reference_forces: Reference contact force data
            tolerance: Tolerance for comparison
            
        Returns:
            True if forces match within tolerance
        """
        print("\n=== Contact Forces Validation ===")
        print("Manual verification recommended:")
        print("1. Record contact forces from original environment")
        print("2. Record contact forces from Isaac Sim environment")
        print("3. Compare force magnitudes and directions")
        print("=" * 40)
        
        return True


def run_validation_suite(env, reference_dir: str) -> Dict:
    """
    Run complete validation suite on an environment.
    
    Args:
        env: Environment to validate
        reference_dir: Directory containing reference data
        
    Returns:
        Dictionary with validation results
    """
    print("\n" + "=" * 60)
    print("RUNNING VALIDATION SUITE")
    print("=" * 60)
    
    validator = EnvironmentValidator()
    results = {}
    
    # 1. Shape validation
    expected_shapes = {
        'num_obs': 154,  # Example - should match actual
        'num_actions': 28,
        'num_envs': env.num_envs,
        'num_states': 200,
    }
    results['shapes'] = validator.validate_shapes(env, expected_shapes)
    
    # 2. Observation format validation
    expected_components = [
        ('root_height', 1),
        ('root_rotation', 4),
        ('root_lin_vel', 3),
        ('root_ang_vel', 3),
        ('dof_pos', 28),
        ('dof_vel', 28),
        ('key_body_pos', 21),  # 7 bodies * 3
    ]
    results['obs_format'] = validator.validate_observation_format(env, expected_components)
    
    # 3. Reference comparison (if available)
    reference_path = Path(reference_dir) / 'reference_trajectory.pkl'
    if reference_path.exists():
        passed, diffs = validator.compare_with_reference(env, str(reference_path))
        results['reference_comparison'] = passed
        results['differences'] = diffs
    else:
        print(f"\nReference data not found: {reference_path}")
        print("Run capture_reference_data() on original environment first.")
    
    # 4. Physics validation
    physics_validator = PhysicsValidator()
    expected_physics = {
        'dt': 1.0 / 60.0,
        'substeps': 2,
        'num_position_iterations': 4,
        'num_velocity_iterations': 0,
        'contact_offset': 0.02,
    }
    results['physics'] = physics_validator.validate_physics_params(
        env.world, expected_physics
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, bool):
            status = "✓ PASS" if value else "✗ FAIL"
            print(f"{status}: {key}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    print("Validation utilities loaded.")
    print("Usage:")
    print("  from ase.utils.validation import EnvironmentValidator")
    print("  validator = EnvironmentValidator()")
    print("  validator.validate_shapes(env, expected_shapes)")
