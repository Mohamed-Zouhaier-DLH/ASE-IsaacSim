"""
Basic import tests to verify the package structure is correct.

These tests don't require Isaac Sim to be installed.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_ase_package_import():
    """Test that the ase package can be imported."""
    import ase
    assert ase.__version__ == '0.1.0'
    print("✓ ase package imported successfully")


def test_utils_import():
    """Test that utilities package can be imported."""
    import ase.utils
    # Note: Individual classes require numpy/torch, so we just check package import
    print("✓ ase.utils package imported successfully")


def test_env_package_import():
    """Test that environment package can be imported."""
    import ase.env
    print("✓ ase.env package imported successfully")


def test_isaac_sim_module_structure():
    """Test Isaac Sim module structure (without importing Isaac Sim)."""
    # Just check the files exist
    base_env_path = Path(__file__).parent.parent / "ase" / "env" / "isaac_sim" / "base_env.py"
    assert base_env_path.exists(), "base_env.py should exist"
    
    wrapper_path = Path(__file__).parent.parent / "ase" / "env" / "isaac_sim" / "compatibility_wrapper.py"
    assert wrapper_path.exists(), "compatibility_wrapper.py should exist"
    
    template_path = Path(__file__).parent.parent / "ase" / "env" / "isaac_sim" / "tasks" / "humanoid_amp_template.py"
    assert template_path.exists(), "humanoid_amp_template.py should exist"
    
    print("✓ Isaac Sim module files exist")


def test_examples_exist():
    """Test that example scripts exist."""
    examples_dir = Path(__file__).parent.parent / "examples"
    
    assert (examples_dir / "convert_assets.py").exists()
    assert (examples_dir / "validate_environment.py").exists()
    assert (examples_dir / "run_isaac_sim_env.py").exists()
    
    print("✓ Example scripts exist")


def test_documentation_exists():
    """Test that documentation files exist."""
    docs_dir = Path(__file__).parent.parent
    
    assert (docs_dir / "README.md").exists()
    assert (docs_dir / "MIGRATION_GUIDE.md").exists()
    assert (docs_dir / "TECHNICAL_PLAN.md").exists()
    assert (docs_dir / "API_MAPPING.md").exists()
    assert (docs_dir / "IMPLEMENTATION_SUMMARY.md").exists()
    
    print("✓ Documentation files exist")


def test_configuration_exists():
    """Test that configuration files exist."""
    root_dir = Path(__file__).parent.parent
    
    assert (root_dir / "requirements.txt").exists()
    assert (root_dir / ".gitignore").exists()
    
    print("✓ Configuration files exist")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running Basic Import Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_ase_package_import,
        test_utils_import,
        test_env_package_import,
        test_isaac_sim_module_structure,
        test_examples_exist,
        test_documentation_exists,
        test_configuration_exists,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    
    if failed == 0:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
