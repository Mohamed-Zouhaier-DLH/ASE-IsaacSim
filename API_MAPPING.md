# API Mapping: Isaac Gym → Isaac Sim

This document provides a detailed mapping between Isaac Gym and Isaac Sim APIs for common operations used in the ASE codebase.

## Table of Contents

1. [Initialization](#initialization)
2. [Asset Loading](#asset-loading)
3. [Environment Creation](#environment-creation)
4. [Physics Configuration](#physics-configuration)
5. [State Access](#state-access)
6. [Action Application](#action-application)
7. [Stepping](#stepping)
8. [Observation Collection](#observation-collection)
9. [Rendering](#rendering)
10. [Cleanup](#cleanup)

---

## Initialization

### Isaac Gym
```python
from isaacgym import gymapi
import torch

gym = gymapi.acquire_gym()
```

### Isaac Sim
```python
from omni.isaac.kit import SimulationApp
from omni.isaac.core import World
import torch

# Create simulation app (must be first)
simulation_app = SimulationApp({"headless": True})

# Create world
world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0/60.0,
    rendering_dt=1.0/60.0,
    backend='torch',
    device='cuda:0'
)
```

**Key Differences:**
- Isaac Sim requires `SimulationApp` initialization first
- `World` object encapsulates scene and physics
- Must specify backend ('torch', 'numpy', or 'warp')

---

## Asset Loading

### Isaac Gym
```python
asset_root = "assets/"
asset_file = "humanoid.xml"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.use_mesh_materials = True
asset_options.flip_visual_attachments = True

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
```

### Isaac Sim
```python
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation

# Load USD asset
usd_path = "assets/humanoid.usd"
prim_path = "/World/Humanoid"

add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

# Create articulation object
articulation = Articulation(prim_path=prim_path)
world.scene.add(articulation)
```

**Key Differences:**
- Isaac Sim uses USD format, not MJCF/URDF
- Must add asset to USD stage first
- Then wrap in Articulation object
- No direct equivalent to `AssetOptions` - configure in USD

---

## Environment Creation

### Isaac Gym
```python
num_envs = 1024
env_spacing = 2.0
num_per_row = int(np.sqrt(num_envs))

envs = []
actor_handles = []

lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

for i in range(num_envs):
    # Create environment
    env = gym.create_env(sim, lower, upper, num_per_row)
    envs.append(env)
    
    # Create actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 1.0)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    
    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 0)
    actor_handles.append(actor_handle)
```

### Isaac Sim
```python
num_envs = 1024
env_spacing = 2.0

articulations = []
rows = int(np.sqrt(num_envs))
cols = int(np.ceil(num_envs / rows))

for i in range(num_envs):
    # Calculate position in grid
    row = i // cols
    col = i % cols
    x = col * env_spacing - (cols - 1) * env_spacing / 2
    y = row * env_spacing - (rows - 1) * env_spacing / 2
    
    # Create prim path for this instance
    prim_path = f"/World/Env_{i}/Humanoid"
    
    # Add reference to stage
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    
    # Create articulation
    articulation = Articulation(prim_path=prim_path)
    articulation.set_world_pose(position=np.array([x, y, 1.0]))
    
    # Add to scene
    world.scene.add(articulation)
    articulations.append(articulation)

# Initialize physics
world.reset()
```

**Key Differences:**
- No separate environment handles in Isaac Sim
- Position articulations manually in grid
- Must call `world.reset()` to initialize physics
- Each articulation is independent USD prim

---

## Physics Configuration

### Isaac Gym
```python
sim_params = gymapi.SimParams()

# Set common parameters
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

# PhysX specific
sim_params.physx.solver_type = 1  # TGS
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.contact_offset = 0.02
sim_params.physx.rest_offset = 0.0
sim_params.physx.bounce_threshold_velocity = 0.2
sim_params.physx.max_depenetration_velocity = 10.0

sim = gym.create_sim(device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
```

### Isaac Sim
```python
from omni.isaac.core.utils.physics import set_physics_scene_parameters

# Configure physics scene
set_physics_scene_parameters(
    physics_dt=1.0/60.0,
    num_position_iterations=4,
    num_velocity_iterations=0,
    contact_offset=0.02,
    rest_offset=0.0,
    bounce_threshold=0.2 * 0.2,  # Note: squared in Isaac Sim
    enable_gpu_dynamics=True,
    use_gpu=True
)

# Gravity is set via World
world.get_physics_context().set_gravity(value=[0.0, 0.0, -9.81])
```

**Key Differences:**
- Physics params set via utility function
- Some parameters named differently
- Gravity set separately via physics context
- Substeps controlled differently (physics_dt vs rendering_dt)

---

## State Access

### Isaac Gym
```python
# Get state tensors (GPU-direct)
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
rigid_body_state_tensor = gym.acquire_rigid_body_state_tensor(sim)

# Refresh tensors
gym.refresh_dof_state_tensor(sim)
gym.refresh_actor_root_state_tensor(sim)
gym.refresh_rigid_body_state_tensor(sim)

# Wrap as torch tensors
dof_state = gymtorch.wrap_tensor(dof_state_tensor)
root_state = gymtorch.wrap_tensor(actor_root_state_tensor)

# Access data
dof_pos = dof_state[:, 0]  # All DOF positions
dof_vel = dof_state[:, 1]  # All DOF velocities
root_pos = root_state[:, 0:3]  # Root positions
root_rot = root_state[:, 3:7]  # Root quaternions
```

### Isaac Sim
```python
# Query each articulation individually
joint_positions_list = []
joint_velocities_list = []
root_positions_list = []
root_orientations_list = []

for articulation in articulations:
    # Joint states
    joint_pos = articulation.get_joint_positions()
    joint_vel = articulation.get_joint_velocities()
    
    # Root state
    root_pos, root_quat = articulation.get_world_pose()
    lin_vel, ang_vel = articulation.get_velocities()
    
    joint_positions_list.append(joint_pos)
    joint_velocities_list.append(joint_vel)
    root_positions_list.append(root_pos)
    root_orientations_list.append(root_quat)

# Stack into batched tensors
dof_pos = torch.stack(joint_positions_list)
dof_vel = torch.stack(joint_velocities_list)
root_pos = torch.stack(root_positions_list)
root_rot = torch.stack(root_orientations_list)
```

**Key Differences:**
- No direct batched tensor access in Isaac Sim
- Must query each articulation individually
- Must stack into batched tensors manually
- No separate refresh calls needed

---

## Action Application

### Isaac Gym
```python
# Set DOF position targets (batched)
gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(actions))

# Or set DOF forces
gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(forces))

# Or set individual actor DOF targets
for i, env in enumerate(envs):
    gym.set_actor_dof_position_targets(env, actor_handles[i], actions[i].cpu().numpy())
```

### Isaac Sim
```python
# Apply position targets to each articulation
for i, articulation in enumerate(articulations):
    articulation.set_joint_position_targets(actions[i])

# Or apply forces
for i, articulation in enumerate(articulations):
    articulation.set_joint_efforts(forces[i])

# Or use action control API
from omni.isaac.core.utils.types import ArticulationAction
for i, articulation in enumerate(articulations):
    action = ArticulationAction(joint_positions=actions[i])
    articulation.apply_action(action)
```

**Key Differences:**
- No batched action application in Isaac Sim
- Must iterate over articulations
- More explicit action types via `ArticulationAction`

---

## Stepping

### Isaac Gym
```python
# Simulate physics
gym.simulate(sim)

# Fetch results (synchronize)
gym.fetch_results(sim, True)

# For graphics
gym.step_graphics(sim)
gym.render_all_camera_sensors(sim)
```

### Isaac Sim
```python
# Step world (physics + rendering if not headless)
world.step(render=False)

# For multiple substeps
for _ in range(substeps):
    world.step(render=False)
```

**Key Differences:**
- Single `world.step()` call for physics + rendering
- No separate fetch or synchronize needed
- Render flag controls whether to render

---

## Observation Collection

### Isaac Gym
```python
# Observations computed from refreshed tensors
def compute_observations():
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    
    # Compute observations from wrapped tensors
    obs = torch.cat([
        root_state[:, 2:3],      # Height
        root_state[:, 3:7],      # Orientation
        root_state[:, 7:10],     # Linear velocity
        root_state[:, 10:13],    # Angular velocity
        dof_state[:, 0],         # Joint positions
        dof_state[:, 1],         # Joint velocities
    ], dim=-1)
    
    return obs
```

### Isaac Sim
```python
def compute_observations():
    # Query states from all articulations
    observations = []
    
    for articulation in articulations:
        # Root state
        root_pos, root_quat = articulation.get_world_pose()
        lin_vel, ang_vel = articulation.get_velocities()
        
        # Joint state
        joint_pos = articulation.get_joint_positions()
        joint_vel = articulation.get_joint_velocities()
        
        # Concatenate
        obs = torch.cat([
            root_pos[2:3],           # Height
            root_quat,               # Orientation
            lin_vel,                 # Linear velocity
            ang_vel,                 # Angular velocity
            joint_pos,               # Joint positions
            joint_vel,               # Joint velocities
        ], dim=-1)
        
        observations.append(obs)
    
    return torch.stack(observations)
```

**Key Differences:**
- Must query each articulation individually
- No tensor refresh mechanism
- Manual stacking required

---

## Rendering

### Isaac Gym
```python
# Setup viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# In loop
if not headless:
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

# Cleanup
gym.destroy_viewer(viewer)
```

### Isaac Sim
```python
# Rendering handled automatically by SimulationApp
# Control via world.step(render=True/False)

# For headless mode, set in SimulationApp config
simulation_app = SimulationApp({"headless": True})

# In loop
world.step(render=not headless)

# No explicit viewer management needed
```

**Key Differences:**
- Rendering integrated into `world.step()`
- No separate viewer object
- Headless mode set at initialization

---

## Cleanup

### Isaac Gym
```python
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
```

### Isaac Sim
```python
simulation_app.close()
```

**Key Differences:**
- Single close call for SimulationApp
- World cleanup automatic

---

## Summary of Key Differences

1. **Initialization**: Isaac Sim requires SimulationApp first, then World
2. **Batching**: Isaac Gym has GPU-batched tensor operations; Isaac Sim requires manual iteration
3. **Asset Format**: MJCF/URDF in Isaac Gym; USD in Isaac Sim
4. **State Access**: Direct tensor access in Isaac Gym; individual queries in Isaac Sim
5. **Actions**: Batched in Isaac Gym; per-articulation in Isaac Sim
6. **Stepping**: Separate simulate/fetch in Isaac Gym; single step in Isaac Sim

## Performance Implications

- Isaac Gym's batched operations are more efficient for large numbers of environments
- Isaac Sim requires more Python overhead for iteration
- Consider using vectorized operations where possible
- Profile early to understand throughput differences

## Migration Strategy

1. Create wrapper functions that abstract differences
2. Use compatibility layer to maintain interface
3. Batch operations where possible in Python
4. Consider trade-off: code complexity vs performance
5. Benchmark regularly to ensure acceptable performance
