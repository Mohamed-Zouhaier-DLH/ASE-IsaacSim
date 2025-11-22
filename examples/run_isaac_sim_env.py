"""
Example script for running an ASE task in Isaac Sim.

This demonstrates how to create and run an Isaac Sim environment,
either for visualization or for integration with RL training.

Usage:
    # Run with visualization (interactive mode)
    python examples/run_isaac_sim_env.py --headless false --num-envs 4
    
    # Run headless for performance testing
    python examples/run_isaac_sim_env.py --headless true --num-envs 64 --steps 1000
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='Run ASE task in Isaac Sim')
    parser.add_argument('--task', type=str, default='HumanoidAMP',
                       help='Task name (e.g., HumanoidAMP, HumanoidReach)')
    parser.add_argument('--num-envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--headless', type=str, default='false',
                       choices=['true', 'false'],
                       help='Run headless (no rendering)')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of steps to run')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='PyTorch device')
    parser.add_argument('--asset-path', type=str, 
                       default='ase/data/assets/usd/amp_humanoid.usd',
                       help='Path to USD asset')
    
    args = parser.parse_args()
    
    headless = args.headless.lower() == 'true'
    
    print("=" * 60)
    print("ASE Isaac Sim Environment Demo")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Num environments: {args.num_envs}")
    print(f"Headless: {headless}")
    print(f"Device: {args.device}")
    print(f"Steps: {args.steps}")
    print()
    
    # Environment configuration
    cfg = {
        'num_envs': args.num_envs,
        'dt': 1.0 / 60.0,
        'substeps': 2,
        'max_episode_length': 1000,
        'env_spacing': 3.0,
        'asset_path': args.asset_path,
        'num_position_iterations': 4,
        'num_velocity_iterations': 0,
        'contact_offset': 0.02,
        'rest_offset': 0.0,
    }
    
    # Create environment
    try:
        if args.task == 'HumanoidAMP':
            from ase.env.isaac_sim.tasks.humanoid_amp_template import HumanoidAMPIsaacSim
            env = HumanoidAMPIsaacSim(cfg, device=args.device, headless=headless)
        else:
            print(f"Task '{args.task}' not yet implemented.")
            print("Available tasks: HumanoidAMP")
            sys.exit(1)
        
        print("✓ Environment created successfully")
        print(f"  Observation space: {env.num_obs}")
        print(f"  Action space: {env.num_actions}")
        print()
        
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Reset environment
    print("Resetting environment...")
    obs = env.reset()
    print(f"✓ Initial observation shape: {obs.shape}")
    print()
    
    # Run simulation loop
    print(f"Running {args.steps} steps...")
    print("-" * 60)
    
    episode_rewards = torch.zeros(args.num_envs, device=args.device)
    episode_lengths = torch.zeros(args.num_envs, device=args.device)
    
    for step in range(args.steps):
        # Generate random actions (replace with policy for real training)
        actions = torch.randn((args.num_envs, env.num_actions), device=args.device) * 0.1
        
        # Step environment
        obs, rewards, dones, extras = env.step(actions)
        
        # Track statistics
        episode_rewards += rewards
        episode_lengths += 1
        
        # Print progress
        if step % 10 == 0 or step == args.steps - 1:
            mean_reward = rewards.mean().item()
            print(f"Step {step:4d}/{args.steps}: "
                  f"reward={mean_reward:6.3f}, "
                  f"done={dones.sum().item():2d}/{args.num_envs}")
        
        # Check for resets
        if dones.any():
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            for idx in reset_ids:
                if idx < args.num_envs:  # Safety check
                    print(f"  Environment {idx.item()} reset: "
                          f"reward={episode_rewards[idx].item():.2f}, "
                          f"length={episode_lengths[idx].item():.0f}")
                    episode_rewards[idx] = 0
                    episode_lengths[idx] = 0
    
    print("-" * 60)
    print("\nSimulation completed!")
    
    # Print final statistics
    print("\nFinal Statistics:")
    print(f"  Mean episode reward: {episode_rewards.mean().item():.2f}")
    print(f"  Mean episode length: {episode_lengths.mean().item():.1f}")
    
    # Cleanup
    print("\nCleaning up...")
    env.cleanup()
    print("✓ Done!")


if __name__ == '__main__':
    main()
