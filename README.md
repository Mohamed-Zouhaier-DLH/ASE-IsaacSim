# ASE-IsaacSim

Migration of NVIDIA's ASE (Adversarial Skill Embeddings) framework from Isaac Gym to Isaac Sim / Isaac Lab.

This repository provides the infrastructure and tools needed to migrate the [ASE (Adversarial Skill Embeddings)](https://github.com/nv-tlabs/ASE) codebase from Isaac Gym to Isaac Sim, while maintaining compatibility with pretrained models and the existing RL training pipeline.

## Overview

ASE is a framework for learning diverse, physics-based character skills through adversarial motion priors. The original implementation uses Isaac Gym for physics simulation. This migration project enables the use of Isaac Sim / Isaac Lab, which offers:

- More advanced rendering and visualization
- USD-based asset pipeline
- Integration with the broader Omniverse ecosystem
- Active development and support

## Status

✅ **Complete ASE Framework with Isaac Sim Migration** ✅

This repository now contains:
- ✅ **Complete original ASE codebase** (from nv-tlabs/ASE)
  - All 9 task implementations (HumanoidAMP, HumanoidReach, HumanoidStrike, HumanoidHeading, HumanoidLocation, HumanoidPerturb, HumanoidAMPGetup, HumanoidViewMotion, and base Humanoid)
  - Full learning infrastructure (AMP, ASE, HRL agents)
  - Complete poselib for motion retargeting
- ✅ **Motion capture data and pretrained models**
  - Motion clips (.npy files) for walking, running, jogging
  - Reallusion sword & shield motion dataset
  - Pre-trained ASE low-level controller (85MB)
  - Pre-trained high-level controllers for all tasks (19MB each)
- ✅ **Isaac Sim migration infrastructure**
  - Comprehensive migration guides and technical documentation
  - Compatibility wrapper for maintaining RLGPUEnv interface
  - Base environment classes for Isaac Sim implementation
  - Asset conversion utilities (MJCF → USD)
  - Validation and testing utilities
  - Example Isaac Sim implementations

## Documentation

- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Comprehensive migration guide with step-by-step instructions
- **[TECHNICAL_PLAN.md](TECHNICAL_PLAN.md)** - Detailed technical specifications and architecture

## Quick Start

### Prerequisites

1. **Isaac Sim** - Install from [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/)
   ```bash
   # Typical installation location
   ~/.local/share/ov/pkg/isaac_sim-*
   ```

2. **Python Environment** - Use Isaac Sim's Python environment
   ```bash
   source ~/.local/share/ov/pkg/isaac_sim-*/setup_python_env.sh
   ```

3. **Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Installation

```bash
# Clone this repository
git clone https://github.com/Mohamed-Zouhaier-DLH/ASE-IsaacSim.git
cd ASE-IsaacSim

# Install dependencies
pip install -r requirements.txt
```

## Using the Original Isaac Gym Implementation

The complete original ASE codebase is included and can be used directly with Isaac Gym:

### Training with Isaac Gym

Train an ASE model to imitate motion clips:

```bash
# Pre-training ASE low-level controller
python ase/run.py --task HumanoidAMPGetup \
    --cfg_env ase/data/cfg/humanoid_ase_sword_shield_getup.yaml \
    --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml \
    --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
    --headless

# Task-training with high-level controller
python ase/run.py --task HumanoidHeading \
    --cfg_env ase/data/cfg/humanoid_sword_shield_heading.yaml \
    --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml \
    --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy \
    --llc_checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth \
    --headless
```

### Testing Pre-Trained Models

Test the provided pre-trained models:

```bash
# Test ASE low-level controller
python ase/run.py --test --task HumanoidAMPGetup --num_envs 16 \
    --cfg_env ase/data/cfg/humanoid_ase_sword_shield_getup.yaml \
    --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml \
    --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml \
    --checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth

# Test high-level heading controller
python ase/run.py --test --task HumanoidHeading --num_envs 16 \
    --cfg_env ase/data/cfg/humanoid_sword_shield_heading.yaml \
    --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml \
    --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy \
    --llc_checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth \
    --checkpoint ase/data/models/ase_hlc_heading_reallusion_sword_shield.pth
```

### Visualizing Motion Data

View motion clips:

```bash
python ase/run.py --test --task HumanoidViewMotion --num_envs 2 \
    --cfg_env ase/data/cfg/humanoid_sword_shield.yaml \
    --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml \
    --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy
```

### Available Tasks

- **HumanoidAMP** - Adversarial Motion Priors
- **HumanoidAMPGetup** - AMP with get-up from ground
- **HumanoidReach** - Reach to target locations
- **HumanoidStrike** - Strike targets
- **HumanoidHeading** - Navigate towards headings
- **HumanoidLocation** - Navigate to locations
- **HumanoidPerturb** - Robustness to perturbations
- **HumanoidViewMotion** - Visualize motion clips

## Migrating to Isaac Sim

### Asset Conversion

Convert MJCF humanoid models to USD format for Isaac Sim:

```bash
# Convert a single asset
python examples/convert_assets.py \
    --input ase/data/assets/mjcf/amp_humanoid.xml \
    --output ase/data/assets/usd/amp_humanoid.usd

# Or convert all assets at once
python examples/convert_assets.py --batch
```

### Running Isaac Sim Environments

```bash
# Run Isaac Sim environment demo
python examples/run_isaac_sim_env.py \
    --task HumanoidAMP \
    --num-envs 4 \
    --headless false \
    --steps 100
```

### Validation

```bash
# Validate Isaac Sim implementation
python examples/validate_environment.py \
    --mode suite \
    --env isaac_sim \
    --num-envs 4
```

## Architecture

The migration maintains the original RL training interface while replacing simulator-specific components:

```
┌─────────────────────────────────────┐
│     RL Training Pipeline            │
│      (rl_games runner)              │
│       [NO CHANGES]                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Compatibility Wrapper Layer       │
│  (RLGPUEnv → Isaac Sim)             │
│       [NEW COMPONENT]               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Isaac Sim Environment Layer       │
│  (Multi-instance, USD physics)      │
│       [NEW COMPONENT]               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Reusable Components              │
│  (poselib, motions, utilities)      │
│      [MINIMAL CHANGES]              │
└─────────────────────────────────────┘
```

## Project Structure

```
ASE-IsaacSim/
├── ase/
│   ├── env/
│   │   ├── isaac_sim/           # Isaac Sim implementations
│   │   │   ├── base_env.py      # Base environment class
│   │   │   ├── compatibility_wrapper.py  # RLGPUEnv adapter
│   │   │   └── tasks/           # Task implementations
│   │   │       ├── humanoid_amp_template.py
│   │   │       └── ...
│   ├── utils/
│   │   ├── asset_conversion.py  # MJCF → USD conversion
│   │   └── validation.py        # Validation utilities
│   └── data/
│       └── assets/
│           ├── mjcf/            # Original MJCF assets
│           └── usd/             # Converted USD assets
├── examples/
│   ├── convert_assets.py        # Asset conversion script
│   ├── validate_environment.py  # Validation script
│   └── run_isaac_sim_env.py     # Environment demo
├── tests/                       # Test suite
├── MIGRATION_GUIDE.md          # Detailed migration guide
├── TECHNICAL_PLAN.md           # Technical specifications
└── requirements.txt            # Python dependencies
```

## Key Components

### 1. Base Environment (`ase/env/isaac_sim/base_env.py`)

Provides the foundation for all ASE tasks in Isaac Sim:
- World creation and management
- Multi-instance environment setup
- Common observation/action buffer management
- Physics stepping and reset logic

### 2. Compatibility Wrapper (`ase/env/isaac_sim/compatibility_wrapper.py`)

Maintains the RLGPUEnv interface expected by rl_games:
- Translates between rl_games API and Isaac Sim
- Preserves observation/action formats
- Enables drop-in replacement for original environments

### 3. Asset Conversion (`ase/utils/asset_conversion.py`)

Utilities for migrating MJCF assets to USD:
- MJCF parsing and analysis
- USD conversion (via Isaac Sim importer)
- Joint ordering and property validation
- Batch conversion support

### 4. Validation Tools (`ase/utils/validation.py`)

Ensure migration correctness:
- Shape and format validation
- Deterministic trajectory comparison
- Physics parameter verification
- Pretrained policy testing

## Migration Workflow

1. **Baseline Documentation**
   - Record obs/action shapes and physics parameters
   - Document joint ordering and properties
   - Save reference trajectories

2. **Asset Conversion**
   - Convert MJCF humanoid models to USD
   - Validate joint names, limits, and rest poses
   - Verify with poselib skeleton

3. **Single-Instance Environment**
   - Implement basic task in Isaac Sim
   - Match observation/action formats exactly
   - Validate against reference data

4. **Physics Matching**
   - Configure PhysX parameters
   - Match joint damping, stiffness, and gains
   - Verify contact behavior

5. **Vectorization**
   - Create multi-instance environments
   - Implement batched step/reset
   - Benchmark performance

6. **Integration**
   - Hook into rl_games training pipeline
   - Test with pretrained models
   - Performance tuning

## Testing

```bash
# Run validation suite
python examples/validate_environment.py --mode suite --env isaac_sim

# Run specific validation
python examples/validate_environment.py --mode validate --env isaac_sim

# Capture reference data (from original Isaac Gym environment)
python examples/validate_environment.py --mode capture --env isaac_gym
```

## Performance Considerations

- **Isaac Gym baseline**: 2048-4096 parallel environments typical for training
- **Isaac Sim target**: Start with 512-1024 environments, optimize as needed
- **Physics tuning**: Adjust solver iterations, substeps for throughput
- **Instancing**: Use USD instancing for better memory efficiency
- **Headless mode**: Always use headless mode for training

## Known Limitations

1. **Asset Conversion**: Manual validation required for complex assets
2. **Physics Differences**: PhysX parameters may not match exactly
3. **Performance**: May not achieve same throughput as Isaac Gym initially
4. **API Changes**: Isaac Sim API evolves; code may need updates

## Contributing

Contributions are welcome! Areas needing work:

- Complete task implementations (humanoid_reach, humanoid_strike, etc.)
- Motion library integration
- AMP discriminator implementation
- Performance optimization
- Additional validation tests

## Resources

- **Original ASE Repository**: https://github.com/nv-tlabs/ASE
- **ASE Paper**: https://xbpeng.github.io/projects/ASE/
- **Isaac Sim Documentation**: https://docs.omniverse.nvidia.com/isaacsim/
- **Isaac Lab Documentation**: https://isaac-sim.github.io/IsaacLab/

## Citation

If you use this work, please cite the original ASE paper:

```bibtex
@article{peng2022ase,
  title={ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters},
  author={Peng, Xue Bin and Guo, Yunrong and Halper, Lina and Levine, Sergey and Fidler, Sanja},
  journal={ACM Trans. Graph.},
  year={2022}
}
```

## License

This migration infrastructure is provided as-is for educational and research purposes. The original ASE code and assets are subject to their respective licenses from nv-tlabs/ASE.

## Contact

For questions or issues related to this migration:
- Open an issue on GitHub
- Refer to Isaac Sim forums for Isaac Sim-specific questions

---

**Note**: This is a migration infrastructure project. To use it, you'll need to:
1. Obtain the original ASE codebase and data from nv-tlabs/ASE
2. Install Isaac Sim from NVIDIA Omniverse
3. Follow the migration guide to integrate the components
