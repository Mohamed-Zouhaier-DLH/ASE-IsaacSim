"""
Isaac Sim task implementations for ASE.

This module contains Isaac Sim implementations of all ASE tasks:
- HumanoidAMP: Base adversarial motion priors task
- HumanoidReach: Reaching task
- HumanoidStrike: Striking task
- HumanoidHeading: Heading control task
- HumanoidLocation: Location navigation task
- HumanoidPerturb: Perturbation robustness task
- HumanoidAMPGetup: Get-up from ground task
- HumanoidViewMotion: Motion visualization task
"""

# Task registry for factory creation
TASK_REGISTRY = {}

# Tasks will be added to registry as they are implemented
# from .humanoid_amp import HumanoidAMPIsaacSim
# TASK_REGISTRY['HumanoidAMP'] = HumanoidAMPIsaacSim

__all__ = ['TASK_REGISTRY']
