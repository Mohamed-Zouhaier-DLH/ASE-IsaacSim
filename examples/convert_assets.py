"""
Example script for converting ASE MJCF assets to USD format.

This script demonstrates how to use the asset conversion utilities to
migrate MJCF humanoid models to USD for use in Isaac Sim.

Usage:
    # Convert a single asset
    python examples/convert_assets.py --input ase/data/assets/mjcf/amp_humanoid.xml \
                                      --output ase/data/assets/usd/amp_humanoid.usd
    
    # Convert all assets in a directory
    python examples/convert_assets.py --batch
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ase.utils.asset_conversion import MJCFToUSDConverter, convert_ase_assets


def main():
    parser = argparse.ArgumentParser(description='Convert MJCF assets to USD')
    parser.add_argument('--input', type=str, help='Input MJCF file path')
    parser.add_argument('--output', type=str, help='Output USD file path')
    parser.add_argument('--batch', action='store_true', help='Convert all assets')
    parser.add_argument('--validate', action='store_true', default=True, 
                       help='Validate conversion (default: True)')
    parser.add_argument('--mjcf-dir', type=str, default='ase/data/assets/mjcf',
                       help='Directory containing MJCF files (for batch mode)')
    parser.add_argument('--usd-dir', type=str, default='ase/data/assets/usd',
                       help='Output directory for USD files (for batch mode)')
    
    args = parser.parse_args()
    
    if args.batch:
        print("=" * 60)
        print("Batch Asset Conversion: MJCF → USD")
        print("=" * 60)
        convert_ase_assets(args.mjcf_dir, args.usd_dir)
    elif args.input and args.output:
        print("=" * 60)
        print("Single Asset Conversion: MJCF → USD")
        print("=" * 60)
        print(f"Input:  {args.input}")
        print(f"Output: {args.output}")
        print()
        
        converter = MJCFToUSDConverter()
        success = converter.convert(args.input, args.output, validate=args.validate)
        
        if success:
            print("\n✓ Conversion successful!")
            
            # Generate validation report
            report_path = args.output.replace('.usd', '_validation.json')
            converter.generate_validation_report(args.input, args.output, report_path)
            
            sys.exit(0)
        else:
            print("\n✗ Conversion failed!")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
