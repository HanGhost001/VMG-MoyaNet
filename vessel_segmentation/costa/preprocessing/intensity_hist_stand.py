# This script aims to perform histogram standardization of Multi-site MRA (MsMRA) images
# Please note that we use a large scale unlabeled multi-site (or various manufacturers) MRA scans,
# in order to learn the global histogram representation of different MRA scans.
# We use torchio to perform histogram standardization

import argparse
import glob
import os

import torch
import torchio
import tqdm
import SimpleITK as sitk


def get_default_landmarks_path() -> str:
    """Get the default landmarks.pth path relative to this file."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "landmarks.pth")


def perform_histogram_standardization(in_folder, landmarks, out_folder):
    """
    Perform histogram standardization on all NIfTI files in a folder.

    Args:
        in_folder: Input folder containing .nii.gz files
        landmarks: Path to landmarks.pth file
        out_folder: Output folder for normalized files
    """
    # Load landmarks with weights_only=False for PyTorch 2.6+ compatibility
    landmarks_dict = torch.load(landmarks, map_location='cpu', weights_only=False)
    for file in tqdm.tqdm(glob.glob(os.path.join(in_folder, "*.nii.gz"))):
        try:
            transform = torchio.HistogramStandardization(landmarks_dict, masking_method=lambda x: x > 0)
            subject = torchio.Subject(
                image=torchio.ScalarImage(file)
            )
            transformed = transform(subject)
            transformed_image = transformed['image'].as_sitk()
            sitk.WriteImage(transformed_image, os.path.join(out_folder, os.path.basename(file)))
        except Exception as e:
            print(f"Warning: Failed to process {os.path.basename(file)}: {e}")


def plan_the_costa_input_dir(raw_dir, normed_dir):
    """
    Create COSTA input format: dual-channel (raw + normalized) per patient.

    Args:
        raw_dir: Directory containing skull-stripped raw MRA images
        normed_dir: Directory containing histogram-standardized images
    """
    import shutil

    raw_dir = os.path.abspath(raw_dir)
    normed_dir = os.path.abspath(normed_dir)
    if not os.path.exists(normed_dir):
        raise ValueError("Histogram standardized folder does not exist")

    output_dir = os.path.join(os.path.dirname(raw_dir), "costa_inputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_num = 0
    for file in tqdm.tqdm(glob.glob(os.path.join(raw_dir, "*.nii.gz"))):
        file_basename = os.path.basename(file)[:-7]
        normed_file = os.path.join(normed_dir, os.path.basename(file))
        dst_raw_file = os.path.join(output_dir, file_basename + "_0000.nii.gz")
        dst_normed_file = os.path.join(output_dir, file_basename + "_0001.nii.gz")
        shutil.copyfile(src=file, dst=dst_raw_file)
        shutil.copyfile(src=normed_file, dst=dst_normed_file)
        file_num += 1

    print(f"Done! Total {file_num} files copied to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Histogram standardization for MRA images")
    parser.add_argument("-i", "--input_folder", required=True, help="Input folder of MRA files")
    parser.add_argument("-o", "--output_folder", required=False, default=None,
                        help="Output folder (default: <input>_normed)")
    parser.add_argument("-l", "--landmarks", required=False, default=None,
                        help="Path to landmarks.pth (default: bundled landmarks)")

    args = parser.parse_args()
    input_folder = os.path.abspath(args.input_folder)

    if len(os.listdir(input_folder)) <= 0:
        raise Exception("The input folder is empty")

    if args.output_folder is None:
        output_folder = input_folder + "_normed"
    else:
        output_folder = os.path.abspath(args.output_folder)

    os.makedirs(output_folder, exist_ok=True)

    landmarks_path = args.landmarks or get_default_landmarks_path()
    if not os.path.exists(landmarks_path):
        raise FileNotFoundError(f"Landmarks file not found: {landmarks_path}")

    print(f"Performing histogram standardization using: {landmarks_path}")
    perform_histogram_standardization(in_folder=input_folder, landmarks=landmarks_path, out_folder=output_folder)


if __name__ == '__main__':
    main()
