"""
Vessel Segmentation Pipeline for MRA Images

This script provides a complete pipeline for brain vessel segmentation:
1. Skull stripping using BET2 (FSL)
2. Histogram standardization
3. COSTA model inference

Usage:
    python segment_vessels.py --input_dir <input_mra_folder> --output_dir <output_folder>

Requirements:
    - FSL installed (for BET2 skull stripping)
    - COSTA model weights
    - nnUNet framework
"""

import os
import sys
import glob
import shutil
import argparse
from pathlib import Path

import torch
import torchio
import SimpleITK as sitk
from tqdm import tqdm


def load_landmarks(landmarks_path: str) -> dict:
    """Load histogram standardization landmarks."""
    if not os.path.exists(landmarks_path):
        raise FileNotFoundError(f"Landmarks file not found: {landmarks_path}")
    return torch.load(landmarks_path, map_location='cpu', weights_only=False)


def perform_skull_stripping(input_dir: str, output_dir: str, f: float = 0.04, g: float = 0.0) -> None:
    """
    Perform skull stripping using BET2 from FSL.
    
    Args:
        input_dir: Directory containing input MRA images
        output_dir: Directory to save skull-stripped images
        f: Fractional intensity threshold (0->1); default=0.04
        g: Vertical gradient in fractional intensity threshold (-1->1); default=0
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("[Step 1] Skull Stripping using BET2...")
    print("Note: FSL must be installed. See: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation")
    
    files = glob.glob(os.path.join(input_dir, "*.nii.gz"))
    for file in tqdm(files, desc="Skull stripping"):
        output_file = os.path.join(output_dir, os.path.basename(file))
        cmd = f'bet2 "{file}" "{output_file}" -f {f} -g {g}'
        os.system(cmd)


def perform_histogram_standardization(input_dir: str, output_dir: str, landmarks_path: str) -> None:
    """
    Perform histogram standardization for intensity normalization.
    
    Args:
        input_dir: Directory containing skull-stripped MRA images
        output_dir: Directory to save normalized images
        landmarks_path: Path to landmarks.pth file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("[Step 2] Histogram Standardization...")
    landmarks_dict = load_landmarks(landmarks_path)
    
    files = glob.glob(os.path.join(input_dir, "*.nii.gz"))
    for file in tqdm(files, desc="Normalizing"):
        try:
            transform = torchio.HistogramStandardization(
                landmarks_dict, 
                masking_method=lambda x: x > 0
            )
            subject = torchio.Subject(image=torchio.ScalarImage(file))
            transformed = transform(subject)
            transformed_image = transformed['image'].as_sitk()
            output_file = os.path.join(output_dir, os.path.basename(file))
            sitk.WriteImage(transformed_image, output_file)
        except Exception as e:
            print(f"Warning: Failed to process {os.path.basename(file)}: {e}")


def create_costa_inputs(raw_dir: str, normed_dir: str, output_dir: str) -> None:
    """
    Create COSTA input format (dual-channel: raw + normalized).
    
    Args:
        raw_dir: Directory containing skull-stripped raw images
        normed_dir: Directory containing normalized images
        output_dir: Directory to save COSTA-format inputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("[Step 3] Creating COSTA Input Format...")
    files = glob.glob(os.path.join(raw_dir, "*.nii.gz"))
    
    for file in tqdm(files, desc="Creating inputs"):
        basename = os.path.basename(file)[:-7]  # Remove .nii.gz
        normed_file = os.path.join(normed_dir, os.path.basename(file))
        
        if os.path.exists(normed_file):
            dst_raw = os.path.join(output_dir, f"{basename}_0000.nii.gz")
            dst_normed = os.path.join(output_dir, f"{basename}_0001.nii.gz")
            shutil.copyfile(file, dst_raw)
            shutil.copyfile(normed_file, dst_normed)


def run_costa_inference(input_dir: str, output_dir: str, model_path: str, 
                        folds: tuple = (0,), checkpoint: str = "model_best") -> None:
    """
    Run COSTA model inference for vessel segmentation.
    
    Args:
        input_dir: Directory containing COSTA-format inputs
        output_dir: Directory to save segmentation results
        model_path: Path to COSTA trained model folder
        folds: Which folds to use for ensemble (default: fold 0)
        checkpoint: Which checkpoint to use (default: model_best)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("[Step 4] Running COSTA Inference...")
    
    # Build the command for COSTA prediction
    folds_str = " ".join(str(f) for f in folds)
    cmd = (
        f'COSTA_predict '
        f'-i "{input_dir}" '
        f'-o "{output_dir}" '
        f'-t 99 '
        f'-tr COSTA '
        f'-m CESAR '
        f'-f {folds_str} '
        f'-chk {checkpoint} '
        f'--num_threads_preprocessing 1 '
        f'--num_threads_nifti_save 1'
    )
    
    print(f"Command: {cmd}")
    os.system(cmd)


def run_vessel_segmentation(
    input_dir: str,
    output_dir: str,
    model_path: str = None,
    landmarks_path: str = None,
    skip_skull_strip: bool = False,
    bet_f: float = 0.04,
    bet_g: float = 0.0
) -> None:
    """
    Complete vessel segmentation pipeline.
    
    Args:
        input_dir: Directory containing input MRA images (.nii.gz)
        output_dir: Directory to save all outputs
        model_path: Path to COSTA model (optional, uses default if not specified)
        landmarks_path: Path to histogram landmarks (optional, uses default if not specified)
        skip_skull_strip: If True, assumes input is already skull-stripped
        bet_f: BET2 fractional intensity threshold
        bet_g: BET2 vertical gradient
    """
    # Set default paths
    script_dir = Path(__file__).parent
    if landmarks_path is None:
        landmarks_path = script_dir / "costa" / "preprocessing" / "landmarks.pth"
    
    # Create output directories
    skull_stripped_dir = os.path.join(output_dir, "skull_stripped")
    normalized_dir = os.path.join(output_dir, "normalized")
    costa_inputs_dir = os.path.join(output_dir, "costa_inputs")
    segmentation_dir = os.path.join(output_dir, "segmentation_results")
    
    # Step 1: Skull stripping
    if not skip_skull_strip:
        perform_skull_stripping(input_dir, skull_stripped_dir, bet_f, bet_g)
        raw_dir = skull_stripped_dir
    else:
        raw_dir = input_dir
        print("[Step 1] Skipped - using pre-skull-stripped images")
    
    # Step 2: Histogram standardization
    perform_histogram_standardization(raw_dir, normalized_dir, str(landmarks_path))
    
    # Step 3: Create COSTA inputs
    create_costa_inputs(raw_dir, normalized_dir, costa_inputs_dir)
    
    # Step 4: Run COSTA inference (if model path provided)
    if model_path:
        run_costa_inference(costa_inputs_dir, segmentation_dir, model_path)
    else:
        print("\n" + "=" * 60)
        print("Preprocessing Complete!")
        print("=" * 60)
        print(f"COSTA inputs ready at: {costa_inputs_dir}")
        print(f"\nTo run segmentation, use:")
        print(f'COSTA_predict -i "{costa_inputs_dir}" -o "{segmentation_dir}" '
              f'-t 99 -tr COSTA -m CESAR -f 0 -chk model_best')


def main():
    parser = argparse.ArgumentParser(
        description="Brain vessel segmentation pipeline using COSTA model"
    )
    parser.add_argument(
        "-i", "--input_dir", 
        required=True,
        help="Directory containing input MRA images (.nii.gz)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Directory to save outputs"
    )
    parser.add_argument(
        "-m", "--model_path",
        default=None,
        help="Path to COSTA model folder (optional)"
    )
    parser.add_argument(
        "-l", "--landmarks_path",
        default=None,
        help="Path to histogram landmarks.pth (optional, uses default)"
    )
    parser.add_argument(
        "--skip_skull_strip",
        action="store_true",
        help="Skip skull stripping (if input is already skull-stripped)"
    )
    parser.add_argument(
        "--bet_f",
        type=float,
        default=0.04,
        help="BET2 fractional intensity threshold (default: 0.04)"
    )
    parser.add_argument(
        "--bet_g",
        type=float,
        default=0.0,
        help="BET2 vertical gradient (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    run_vessel_segmentation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        landmarks_path=args.landmarks_path,
        skip_skull_strip=args.skip_skull_strip,
        bet_f=args.bet_f,
        bet_g=args.bet_g
    )


if __name__ == "__main__":
    main()
