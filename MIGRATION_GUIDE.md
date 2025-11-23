# ASE Migration Guide: Isaac Gym → Isaac Sim / Isaac Lab

This guide provides a comprehensive plan for migrating the ASE (Adversarial Skill Embeddings) codebase from Isaac Gym to Isaac Sim / Isaac Lab.

## Overview

The migration involves replacing simulator-specific components while preserving the environment interface, observation/action formats, and compatibility with pretrained models.

## Quick Inventory

### Files That Need Migration

1. **Environment Implementations** (`ase/env/tasks/*.py`)
   - `humanoid.py` - Base humanoid task
   - `humanoid_amp.py` - AMP (Adversarial Motion Priors) task
   - `humanoid_amp_getup.py` - Get-up from ground task
   - `humanoid_heading.py` - Heading control task
   - `humanoid_location.py` - Location navigation task
   - `humanoid_perturb.py` - Perturbation robustness task
   - `humanoid_reach.py` - Reaching task
   - `humanoid_strike.py` - Striking task
   - `humanoid_view_motion.py` - Motion playback/visualization
   - `vec_task.py` and `vec_task_wrappers.py` - Vectorized environment wrappers

2. **Assets** (`ase/data/assets/mjcf`)
   - MJCF humanoid models that need USD conversion
   - Joint ordering and naming must be preserved

3. **Motion & Pose Pipeline** (`ase/data/motions`, `ase/poselib`)
   - Motion clips (.npy files) - reusable
   - Retargeting utilities - need skeleton mapping validation
   - Poselib utilities - largely reusable

4. **Pretrained Models** (`ase/data/models`)
   - Saved policies requiring exact obs/action format preservation

5. **Runner Code** (`ase/run.py`)
   - Boot/runner expecting RLGPUEnv interface
   - Must provide compatible wrapper

## Major Differences to Handle

### 1. API Differences

**Isaac Gym:**
- GPU-batched API with direct tensor access
- `gymtorch` utilities for tensor manipulation
- Single-call batched stepping

**Isaac Sim:**
- Omniverse/USD-based scene management
- Different articulation creation methods
- Different physics stepping API
- Multi-instance features for parallelism

### 2. Asset Formats

**Isaac Gym:** MJCF native support

**Isaac Sim:** USD native format
- Requires MJCF → USD conversion
- Must preserve:
  - Joint names and ordering
  - Joint limits
  - Actuator mappings
  - Rest poses / T-pose

### 3. Vectorization & Performance

**Isaac Gym:** Optimized for massive GPU-batched parallel sims

**Isaac Sim:** Different multi-instance strategy
- Requires careful tuning of:
  - Physics solver parameters
  - Substeps
  - Multi-stage instancing
- Different performance characteristics

### 4. Tensor Handling

**Isaac Gym:** Direct gymtorch tensor buffers

**Isaac Sim:** Manual tensor mapping
- Extract data from Isaac Sim
- Map to PyTorch tensors on CUDA
- Feed actions back to sim

## Practical Migration Plan

### Step 1: Baseline & Documentation
- [ ] Record obs/action shapes from original code
- [ ] Document normalization procedures
- [ ] Document physics parameters (timestep, substeps, solver iterations)
- [ ] Document joint ordering
- [ ] Save example observations and rollouts

### Step 2: Prototype Single-Instance Environment
- [ ] Convert MJCF humanoid → USD
- [ ] Create basic Isaac Sim environment script
- [ ] Implement `step()`, `reset()`, `get_obs()`, `get_state()`
- [ ] Verify shape/ordering matches original
- [ ] Test single-step forward pass

### Step 3: Match Physics & Actuators
- [ ] Configure PhysX parameters
- [ ] Set joint damping
- [ ] Configure drive modes
- [ ] Set motor gains
- [ ] Configure contact settings
- [ ] Validate against original physics

### Step 4: Reuse Poselib & Motion Loaders
- [ ] Keep `ase/poselib` unchanged
- [ ] Keep motion .npy files
- [ ] Adapt motion-apply code for Isaac Sim skeleton
- [ ] Verify joint name → index mapping

### Step 5: Implement Vectorization
- [ ] Create multiple environment instances
- [ ] Implement batched step/reset logic
- [ ] Validate batched observation arrays
- [ ] Benchmark performance

### Step 6: Hook into RL Runner
- [ ] Create compatibility wrapper
- [ ] Implement vecenv interface
- [ ] Preserve `get_state()`, `action_space`, `observation_space` APIs
- [ ] Test with rl_games runner

### Step 7: Test Pretrained Models
- [ ] Load existing checkpoint
- [ ] Run in new environment
- [ ] Debug any behavior differences
- [ ] Verify obs ordering and normalization

### Step 8: Performance Tuning
- [ ] Profile step time
- [ ] Optimize physics solver
- [ ] Tune substeps
- [ ] Enable GPU-accelerated physics
- [ ] Disable rendering during training
- [ ] Benchmark against original

## Risks and Pitfalls

### Asset Conversion Errors
- **Risk:** Joint name mismatches or different rest poses
- **Mitigation:** Verify T-pose and joint order early using poselib

### Observation/Action Mismatch
- **Risk:** Changes in concatenation order, normalization, or coordinate frames
- **Mitigation:** Keep format identical; validate against saved examples

### Performance Drop
- **Risk:** Lower GPU-batched throughput
- **Mitigation:** Reduce parallelism if needed; optimize instancing

### Engineering Effort
- **Risk:** Significant time investment (weeks for production-level)
- **Mitigation:** Start with single-instance validation; scale incrementally

## Tips and Shortcuts

1. **Keep Original Interface:** Implement compatibility wrapper to avoid changing RL/training code

2. **Asset Conversion:** Convert MJCF to USD once and verify thoroughly

3. **Deterministic Testing:** Run trained policies single-instance first

4. **Headless Training:** Use headless mode for training; Lab UI for debugging only

5. **Incremental Validation:** Test each component before moving to next step

## API Mapping Reference

### Environment Creation

**Isaac Gym:**
```python
gym = gymapi.acquire_gym()
sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
env_handle = gym.create_env(sim, lower, upper, num_per_row)
```

**Isaac Sim:**
```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})
from omni.isaac.core import World
world = World(stage_units_in_meters=1.0)
# Use world.scene to add articulations
```

### Asset Loading

**Isaac Gym:**
```python
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
actor_handle = gym.create_actor(env_handle, asset, pose, "actor_name", 0, 0)
```

**Isaac Sim:**
```python
from omni.isaac.core.articulations import Articulation
# Import USD or create articulation programmatically
articulation = Articulation(prim_path="/World/Humanoid")
world.scene.add(articulation)
```

### Physics Stepping

**Isaac Gym:**
```python
gym.simulate(sim)
gym.fetch_results(sim, True)
```

**Isaac Sim:**
```python
world.step(render=False)
```

### State Access

**Isaac Gym:**
```python
dof_states = gym.acquire_dof_state_tensor(sim)
root_states = gym.acquire_actor_root_state_tensor(sim)
# Direct GPU tensor access via gymtorch
```

**Isaac Sim:**
```python
# Query articulation for joint states
joint_positions = articulation.get_joint_positions()
joint_velocities = articulation.get_joint_velocities()
# Convert to torch tensors manually
```

### Action Application

**Isaac Gym:**
```python
gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(actions))
```

**Isaac Sim:**
```python
articulation.set_joint_position_targets(actions)
# or articulation.apply_action(action_control)
```

## File Structure

```
ase/
├── env/
│   ├── tasks/              # Task implementations (NEED MIGRATION)
│   │   ├── humanoid.py
│   │   ├── humanoid_amp.py
│   │   └── ...
│   └── isaac_sim/          # NEW: Isaac Sim implementations
│       ├── base_env.py
│       ├── compatibility_wrapper.py
│       └── tasks/
├── data/
│   ├── assets/
│   │   ├── mjcf/          # Original MJCF assets
│   │   └── usd/           # NEW: Converted USD assets
│   ├── motions/           # Motion clips (REUSABLE)
│   ├── models/            # Pretrained models (PRESERVE COMPATIBILITY)
│   └── ...
├── poselib/               # Pose utilities (LARGELY REUSABLE)
├── utils/
│   └── asset_conversion/  # NEW: MJCF → USD conversion tools
├── run.py                 # Runner (NEEDS WRAPPER)
└── requirements.txt       # NEEDS UPDATE
```

## Next Steps

1. Review this guide thoroughly
2. Set up Isaac Sim development environment
3. Begin with Step 1: Baseline documentation
4. Follow steps sequentially
5. Validate at each stage before proceeding

## Resources

- Isaac Sim Documentation: https://docs.omniverse.nvidia.com/isaacsim/
- Isaac Lab Documentation: https://isaac-sim.github.io/IsaacLab/
- Original ASE Paper: https://xbpeng.github.io/projects/ASE/
- Original ASE Repository: https://github.com/nv-tlabs/ASE

## Support

For issues specific to this migration, refer to:
- Isaac Sim forums: https://forums.developer.nvidia.com/c/omniverse/simulation/69
- Isaac Lab GitHub: https://github.com/isaac-sim/IsaacLab
