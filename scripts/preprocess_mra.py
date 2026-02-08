"""
MRA Preprocessing Script

This script provides a complete preprocessing pipeline for MRA images:
1. Skull stripping (using BET2)
2. Histogram standardization
3. COSTA vessel segmentation
4. Data preparation for training

Usage:
    python preprocess_mra.py --input_dir <raw_mra_folder> --output_dir <output_folder>
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vessel_segmentation.segment_vessels import run_vessel_segmentation


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MRA images for VMG-MoyaNet"
    )
    parser.add_argument(
        "-i", "--input_dir",
        required=True,
        help="Directory containing raw MRA images (.nii.gz)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Directory to save preprocessed outputs"
    )
    parser.add_argument(
        "-m", "--model_path",
        default=None,
        help="Path to COSTA model folder (optional)"
    )
    parser.add_argument(
        "--skip_skull_strip",
        action="store_true",
        help="Skip skull stripping if already done"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VMG-MoyaNet MRA Preprocessing Pipeline")
    print("=" * 60)
    
    run_vessel_segmentation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        skip_skull_strip=args.skip_skull_strip
    )
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. For Houkin grading: use scripts/build_hemi_dataset.py")
    print(f"2. For differential diagnosis: use full-brain data directly")


if __name__ == "__main__":
    main()
