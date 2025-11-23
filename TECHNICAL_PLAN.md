# Technical Migration Plan: ASE to Isaac Sim

## Executive Summary

This document provides detailed technical specifications for migrating the ASE codebase from Isaac Gym to Isaac Sim/Isaac Lab. The migration preserves the RL training interface while replacing simulator-specific implementations.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   RL Training Pipeline                   │
│                  (rl_games runner)                       │
│                    [NO CHANGES]                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Compatibility Wrapper Layer                 │
│         (Translates RLGPUEnv to Isaac Sim)              │
│                    [NEW COMPONENT]                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│            Isaac Sim Environment Layer                   │
│        (Multi-instance, USD-based physics)              │
│                    [NEW COMPONENT]                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Reusable Components                         │
│      (poselib, motions, observation utils)              │
│                  [MINIMAL CHANGES]                       │
└─────────────────────────────────────────────────────────┘
```

## Component-by-Component Analysis

### 1. Environment Tasks (ase/env/tasks/*.py)

#### Current Implementation (Isaac Gym)
```python
class HumanoidAMP(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # Uses gymapi to create sim
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(...)
        
    def create_sim(self):
        # Load MJCF asset
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
    def pre_physics_step(self, actions):
        # Direct tensor manipulation
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(actions))
        
    def compute_observations(self):
        # Direct GPU tensor access
        self.obs_buf[:] = torch.cat([root_states, dof_pos, dof_vel, ...], dim=-1)
```

#### New Implementation (Isaac Sim)
```python
class HumanoidAMPIsaacSim(BaseTaskIsaacSim):
    def __init__(self, cfg, device_type, device_id, headless):
        # Use Isaac Sim World API
        from omni.isaac.core import World
        self.world = World(stage_units_in_meters=1.0, device=device_id)
        
    def create_sim(self):
        # Load USD asset
        from omni.isaac.core.utils.stage import add_reference_to_stage
        add_reference_to_stage(usd_path="/path/to/humanoid.usd", prim_path=f"/World/Humanoid_{i}")
        self.articulations = [Articulation(prim_path=f"/World/Humanoid_{i}") for i in range(self.num_envs)]
        
    def pre_physics_step(self, actions):
        # Batch apply actions to all articulations
        for i, articulation in enumerate(self.articulations):
            articulation.set_joint_position_targets(actions[i])
        
    def compute_observations(self):
        # Query states and convert to tensors
        root_states = torch.stack([art.get_world_pose() for art in self.articulations])
        dof_pos = torch.stack([art.get_joint_positions() for art in self.articulations])
        self.obs_buf[:] = torch.cat([root_states, dof_pos, dof_vel, ...], dim=-1)
```

#### Migration Strategy
1. Create base class `BaseTaskIsaacSim` with common Isaac Sim patterns
2. Subclass for each task (HumanoidAMP, HumanoidReach, etc.)
3. Preserve exact observation/action buffer shapes and ordering
4. Use compatibility layer to maintain RLGPUEnv interface

### 2. Asset Pipeline (ase/data/assets/)

#### Current Assets
- MJCF format humanoid models in `ase/data/assets/mjcf/`
- Contains: `amp_humanoid.xml`, `amp_humanoid_sword_shield.xml`

#### Required Changes
1. **Convert MJCF to USD**
   ```bash
   # Use Isaac Sim's importer
   python convert_mjcf_to_usd.py --input assets/mjcf/amp_humanoid.xml --output assets/usd/amp_humanoid.usd
   ```

2. **Validation Requirements**
   - Joint names must match exactly
   - Joint order must be identical
   - Rest pose / T-pose must match
   - Joint limits must be preserved
   - Actuator properties must match (damping, stiffness, max force)

3. **Verification Script**
   ```python
   # Compare MJCF and USD joint properties
   mjcf_joints = parse_mjcf_joints("amp_humanoid.xml")
   usd_joints = parse_usd_joints("amp_humanoid.usd")
   assert mjcf_joints.keys() == usd_joints.keys()
   for name in mjcf_joints:
       assert_close(mjcf_joints[name].limits, usd_joints[name].limits)
   ```

### 3. Compatibility Wrapper (NEW)

The wrapper translates between rl_games expectations and Isaac Sim implementation.

```python
class IsaacSimVecEnvWrapper:
    """
    Wrapper that provides RLGPUEnv interface for Isaac Sim environments.
    
    This maintains compatibility with the existing rl_games training pipeline
    without requiring changes to the runner code.
    """
    
    def __init__(self, isaac_sim_env):
        self.env = isaac_sim_env
        self.num_envs = isaac_sim_env.num_envs
        self.num_obs = isaac_sim_env.num_obs
        self.num_actions = isaac_sim_env.num_actions
        self.device = isaac_sim_env.device
        
        # Initialize observation and state buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.state_buf = torch.zeros((self.num_envs, self.num_states), device=self.device)
        
    def step(self, actions):
        """
        Step all environments with the given actions.
        
        Returns:
            obs_buf: Observations (num_envs, num_obs)
            reward_buf: Rewards (num_envs,)
            reset_buf: Reset flags (num_envs,)
            extras: Dictionary of additional info
        """
        self.env.pre_physics_step(actions)
        self.env.world.step(render=False)
        self.env.post_physics_step()
        
        return self.env.obs_buf, self.env.reward_buf, self.env.reset_buf, self.env.extras
        
    def reset(self):
        """Reset all environments."""
        self.env.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.env.obs_buf
        
    def get_state(self):
        """Get privileged state information."""
        return self.env.state_buf
```

### 4. Physics Configuration

Critical physics parameters that must match:

```python
# Isaac Gym configuration
sim_params = {
    "dt": 1.0 / 60.0,              # Timestep
    "substeps": 2,                  # Physics substeps per step
    "gravity": [0.0, 0.0, -9.81],
    "physx": {
        "solver_type": 1,           # TGS solver
        "num_position_iterations": 4,
        "num_velocity_iterations": 0,
        "contact_offset": 0.02,
        "rest_offset": 0.0,
        "bounce_threshold_velocity": 0.2,
        "max_depenetration_velocity": 10.0,
        "default_buffer_size_multiplier": 5.0,
    }
}

# Isaac Sim equivalent
from omni.isaac.core.utils.physics import set_physics_scene_parameters
set_physics_scene_parameters(
    physics_dt=1.0/60.0,
    num_position_iterations=4,
    num_velocity_iterations=0,
    contact_offset=0.02,
    rest_offset=0.0,
)
```

### 5. Observation/Action Format Preservation

**Critical Requirement:** Pretrained models expect specific tensor layouts.

#### Observation Format (Humanoid AMP)
```python
# Order must be preserved exactly:
obs = torch.cat([
    root_pos[:, 2:3],          # Root height (1)
    root_rot,                   # Root rotation quaternion (4)
    root_vel,                   # Root linear velocity (3)
    root_ang_vel,              # Root angular velocity (3)
    dof_pos,                   # Joint positions (num_dof)
    dof_vel,                   # Joint velocities (num_dof)
    key_body_pos,              # Key body positions (num_bodies * 3)
], dim=-1)
```

#### Action Format
```python
# PD targets for each DOF
actions = torch.tensor([...])  # Shape: (num_envs, num_dof)
# Applied as position targets with PD control
```

### 6. Motion and Pose Pipeline

#### Poselib (REUSABLE with minor changes)

```python
# ase/poselib/poselib/core/rotation3d.py - No changes needed
# ase/poselib/poselib/skeleton/skeleton3d.py - No changes needed

# Only change: skeleton joint index mapping validation
def validate_skeleton_compatibility(isaac_sim_skeleton, poselib_skeleton):
    """Ensure joint names map to same indices."""
    for name, idx in poselib_skeleton.joint_names.items():
        assert isaac_sim_skeleton.get_joint_index(name) == idx, \
            f"Joint {name} index mismatch: {isaac_sim_skeleton.get_joint_index(name)} vs {idx}"
```

#### Motion Files (REUSABLE)

Motion .npy files contain:
- Joint rotations (quaternions)
- Root translations
- Joint angular velocities

These can be used as-is after validating skeleton compatibility.

### 7. Multi-Instance Strategy

Isaac Sim provides several approaches for parallelism:

#### Option A: Multiple Articulations in Single Stage
```python
# Create many articulations in one USD stage
for i in range(num_envs):
    prim_path = f"/World/Env_{i}/Humanoid"
    add_reference_to_stage(usd_path, prim_path)
    articulation = Articulation(prim_path)
    world.scene.add(articulation)
```

#### Option B: Instancing (More Efficient)
```python
# Use USD instancing for better performance
from omni.isaac.core.utils.prims import create_prim
prototype_path = "/World/Prototypes/Humanoid"
add_reference_to_stage(usd_path, prototype_path)
for i in range(num_envs):
    create_prim(f"/World/Env_{i}/Humanoid", "Xform", 
                attributes={"instanceable": True},
                ref_path=prototype_path)
```

#### Performance Considerations
- Isaac Gym: Typically 2048-4096 parallel envs for training
- Isaac Sim: May need to start with fewer (512-1024) and optimize
- Benchmark early to understand throughput differences

### 8. Testing & Validation Strategy

#### Level 1: Unit Tests
```python
def test_observation_shape():
    """Verify observation buffer shape matches expected."""
    env = HumanoidAMPIsaacSim(cfg)
    obs = env.reset()
    assert obs.shape == (num_envs, num_obs)

def test_action_application():
    """Verify actions are applied correctly."""
    env = HumanoidAMPIsaacSim(cfg)
    actions = torch.zeros((num_envs, num_actions))
    env.step(actions)
    # Verify joint targets were set
```

#### Level 2: Deterministic Comparison
```python
def test_deterministic_step():
    """Compare single step output with reference."""
    env = HumanoidAMPIsaacSim(cfg)
    obs = env.reset()
    actions = torch.zeros((1, num_actions))
    
    obs_new, reward, done, info = env.step(actions)
    
    # Load reference from Isaac Gym run
    reference = load_reference_data("single_step_reference.pkl")
    assert_close(obs_new, reference["obs"], rtol=1e-3)
    assert_close(reward, reference["reward"], rtol=1e-3)
```

#### Level 3: Rollout Validation
```python
def test_policy_rollout():
    """Run pretrained policy and compare behavior."""
    env = HumanoidAMPIsaacSim(cfg)
    policy = load_pretrained_policy("ase_model.pth")
    
    obs = env.reset()
    total_reward = 0
    for _ in range(1000):
        actions = policy(obs)
        obs, reward, done, info = env.step(actions)
        total_reward += reward.mean()
    
    # Compare with reference rollout
    reference_reward = load_reference_reward("policy_rollout_reference.pkl")
    assert abs(total_reward - reference_reward) < 100  # Reasonable tolerance
```

## Implementation Timeline

### Phase 1: Foundation (Week 1-2)
- [ ] Set up Isaac Sim development environment
- [ ] Create base classes and compatibility wrapper
- [ ] Convert first asset (amp_humanoid.xml → USD)
- [ ] Validate asset conversion

### Phase 2: Single Environment (Week 2-3)
- [ ] Implement HumanoidAMP base task
- [ ] Verify observation/action shapes
- [ ] Match physics parameters
- [ ] Single-instance deterministic test

### Phase 3: Vectorization (Week 3-4)
- [ ] Implement multi-instance creation
- [ ] Batch processing for step/reset
- [ ] Performance benchmarking
- [ ] Optimize bottlenecks

### Phase 4: Full Task Suite (Week 4-6)
- [ ] Implement remaining tasks (reach, strike, location, etc.)
- [ ] Motion playback validation
- [ ] Pretrained model testing
- [ ] End-to-end training test

### Phase 5: Polish & Documentation (Week 6-7)
- [ ] Performance tuning
- [ ] Documentation updates
- [ ] Example scripts
- [ ] Migration guide refinement

## Risk Mitigation

### Risk: Asset Conversion Failures
**Mitigation:**
- Manual inspection of converted USD
- Automated joint validation script
- Visual comparison in Isaac Sim viewer

### Risk: Performance Degradation
**Mitigation:**
- Early benchmarking
- Multiple instancing strategies
- Profile and optimize critical paths
- Document achievable throughput

### Risk: Observation/Action Drift
**Mitigation:**
- Deterministic reference tests
- Policy rollout validation
- Careful tensor shape/dtype checking
- Coordinate frame validation

### Risk: Subtle Physics Differences
**Mitigation:**
- Match all PhysX parameters
- Side-by-side video comparison
- Quantitative trajectory comparison
- Contact force logging

## Success Criteria

1. **Functional Equivalence**
   - All 9 task variants working
   - Observation/action shapes preserved
   - Pretrained policies load and run

2. **Physics Fidelity**
   - Motion playback looks correct
   - Policies achieve >80% of original performance
   - Contact behaviors match

3. **Performance**
   - Achieve ≥50% of Isaac Gym throughput
   - Training converges in similar time
   - Scalable to ≥512 parallel environments

4. **Maintainability**
   - Clean code architecture
   - Comprehensive documentation
   - Example scripts and tests
   - Clear migration path for users

## Appendix: File Modifications Checklist

### New Files to Create
- [ ] `ase/env/isaac_sim/base_env.py`
- [ ] `ase/env/isaac_sim/compatibility_wrapper.py`
- [ ] `ase/env/isaac_sim/tasks/humanoid_amp.py`
- [ ] `ase/env/isaac_sim/tasks/humanoid_reach.py`
- [ ] `ase/env/isaac_sim/tasks/humanoid_strike.py`
- [ ] `ase/env/isaac_sim/tasks/humanoid_heading.py`
- [ ] `ase/env/isaac_sim/tasks/humanoid_location.py`
- [ ] `ase/env/isaac_sim/tasks/humanoid_perturb.py`
- [ ] `ase/env/isaac_sim/tasks/humanoid_amp_getup.py`
- [ ] `ase/env/isaac_sim/tasks/humanoid_view_motion.py`
- [ ] `ase/utils/asset_conversion.py`
- [ ] `ase/utils/validation.py`
- [ ] `tests/test_isaac_sim_env.py`
- [ ] `tests/test_asset_conversion.py`
- [ ] `examples/train_isaac_sim.py`
- [ ] `examples/run_policy_isaac_sim.py`

### Files to Modify
- [ ] `ase/run.py` - Add Isaac Sim environment option
- [ ] `requirements.txt` - Add Isaac Sim dependencies
- [ ] `README.md` - Update installation and usage instructions
- [ ] `ase/poselib/poselib/skeleton/skeleton3d.py` - Add validation helper (minor)

### Files to Keep Unchanged
- ✓ `ase/data/motions/*` - Motion files
- ✓ `ase/data/models/*` - Pretrained models
- ✓ `ase/poselib/poselib/core/*` - Core utilities
- ✓ `ase/learning/*` - RL algorithms (if using rl_games)

## Conclusion

This migration is feasible but requires careful attention to:
1. Preserving exact observation/action formats
2. Matching physics parameters precisely
3. Validating asset conversion thoroughly
4. Testing incrementally at each stage

With proper planning and validation, the migrated codebase will maintain compatibility with pretrained models while leveraging Isaac Sim's advanced features.
