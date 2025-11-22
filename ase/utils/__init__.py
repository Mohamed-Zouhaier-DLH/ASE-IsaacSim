"""
Utility modules for ASE Isaac Sim migration.

This package contains utilities for:
- Asset conversion (MJCF to USD)
- Environment validation
- Performance benchmarking

Note: Classes are imported lazily to avoid requiring dependencies
when just importing the package.
"""

def __getattr__(name):
    """Lazy import to avoid requiring numpy/torch at package import time."""
    if name == 'MJCFToUSDConverter':
        from .asset_conversion import MJCFToUSDConverter
        return MJCFToUSDConverter
    elif name == 'JointOrderValidator':
        from .asset_conversion import JointOrderValidator
        return JointOrderValidator
    elif name == 'EnvironmentValidator':
        from .validation import EnvironmentValidator
        return EnvironmentValidator
    elif name == 'PhysicsValidator':
        from .validation import PhysicsValidator
        return PhysicsValidator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'MJCFToUSDConverter',
    'JointOrderValidator',
    'EnvironmentValidator',
    'PhysicsValidator',
]
