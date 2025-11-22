"""
Example script for validating Isaac Sim environment implementation.

This script demonstrates how to use the validation utilities to ensure
the Isaac Sim implementation matches the original Isaac Gym behavior.

Usage:
    # Capture reference data from original environment (Isaac Gym)
    python examples/validate_environment.py --mode capture --env isaac_gym
    
    # Validate Isaac Sim implementation against reference
    python examples/validate_environment.py --mode validate --env isaac_sim
    
    # Run full validation suite
    python examples/validate_environment.py --mode suite --env isaac_sim
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ase.utils.validation import (
    EnvironmentValidator, 
    PhysicsValidator,
    run_validation_suite
)


def create_environment(env_type: str, cfg: dict):
    """
    Create environment based on type.
    
    Args:
        env_type: 'isaac_gym' or 'isaac_sim'
        cfg: Environment configuration
        
    Returns:
        Environment instance
    """
    if env_type == 'isaac_sim':
        from ase.env.isaac_sim.tasks.humanoid_amp_template import HumanoidAMPIsaacSim
        return HumanoidAMPIsaacSim(cfg, device='cuda:0', headless=True)
    elif env_type == 'isaac_gym':
        # This would import the original Isaac Gym environment
        # from ase.env.tasks.humanoid_amp import HumanoidAMP
        # return HumanoidAMP(cfg, rl_device='cuda:0', sim_device='cuda:0', headless=True)
        raise NotImplementedError("Isaac Gym environment not available in this repository")
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def main():
    parser = argparse.ArgumentParser(description='Validate environment implementation')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['capture', 'validate', 'suite'],
                       help='Validation mode')
    parser.add_argument('--env', type=str, required=True,
                       choices=['isaac_gym', 'isaac_sim'],
                       help='Environment type')
    parser.add_argument('--reference-dir', type=str, default='./reference_data',
                       help='Directory for reference data')
    parser.add_argument('--num-steps', type=int, default=100,
                       help='Number of steps for capture/validation')
    parser.add_argument('--num-envs', type=int, default=4,
                       help='Number of parallel environments')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.reference_dir, exist_ok=True)
    
    # Environment configuration
    cfg = {
        'num_envs': args.num_envs,
        'dt': 1.0 / 60.0,
        'substeps': 2,
        'max_episode_length': 1000,
        'env_spacing': 3.0,
        'asset_path': 'ase/data/assets/usd/amp_humanoid.usd',
    }
    
    print("=" * 60)
    print(f"Environment Validation - Mode: {args.mode}")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Num envs: {cfg['num_envs']}")
    print(f"Reference dir: {args.reference_dir}")
    print()
    
    # Create environment
    try:
        env = create_environment(args.env, cfg)
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        sys.exit(1)
    
    # Run validation based on mode
    validator = EnvironmentValidator()
    
    if args.mode == 'capture':
        print("\nCapturing reference data...")
        reference_path = os.path.join(args.reference_dir, 'reference_trajectory.pkl')
        trajectory = validator.capture_reference_data(
            env,
            num_steps=args.num_steps,
            output_path=reference_path
        )
        print(f"✓ Reference data captured: {reference_path}")
        
    elif args.mode == 'validate':
        print("\nValidating against reference...")
        reference_path = os.path.join(args.reference_dir, 'reference_trajectory.pkl')
        
        if not os.path.exists(reference_path):
            print(f"✗ Reference data not found: {reference_path}")
            print("Run with --mode capture first to create reference data")
            sys.exit(1)
        
        passed, differences = validator.compare_with_reference(
            env,
            reference_path,
            tolerance=1e-3
        )
        
        if passed:
            print("\n✓ Validation PASSED")
            sys.exit(0)
        else:
            print("\n✗ Validation FAILED")
            sys.exit(1)
            
    elif args.mode == 'suite':
        print("\nRunning full validation suite...")
        results = run_validation_suite(env, args.reference_dir)
        
        # Check if all tests passed
        all_passed = all(
            isinstance(v, bool) and v for v in results.values()
            if not isinstance(v, dict)
        )
        
        if all_passed:
            print("\n✓ All validation tests PASSED")
            sys.exit(0)
        else:
            print("\n✗ Some validation tests FAILED")
            sys.exit(1)
    
    # Cleanup
    env.cleanup()


if __name__ == '__main__':
    main()
