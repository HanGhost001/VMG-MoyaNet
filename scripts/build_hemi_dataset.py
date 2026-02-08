"""
Build Hemisphere Dataset for Houkin Grading

This script processes full-brain MRA and vessel mask images into hemisphere-level
data for Houkin grade classification.

Processing pipeline:
1. Read Houkin grade Excel spreadsheet
2. Match MRA and Mask files by patient ID
3. Resample to 1.0mm isotropic spacing
4. Split into hemispheres based on mask centroid
5. Flip right hemisphere for alignment
6. Save hemisphere data and generate manifest CSV

Usage:
    python -m scripts.build_hemi_dataset \\
        --excel_path /path/to/houkin_grade.xlsx \\
        --mra_dir /path/to/mra \\
        --mask_dir /path/to/masks \\
        --output_dir /path/to/output \\
        --file_prefix "SUB-"
"""
import os
import csv
import glob
import argparse
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_to_output


def grade_to_label3(g: int) -> int:
    """Convert Houkin grade (1-4) to 3-class label: G1->0, G2->1, G3/G4->2"""
    g = int(g)
    return g - 1 if g <= 2 else 2


def load_canonical(path: str) -> nib.Nifti1Image:
    nii = nib.load(path)
    return nib.as_closest_canonical(nii)


def resample_iso(nii: nib.Nifti1Image, spacing_mm: Tuple[float, float, float], order: int) -> nib.Nifti1Image:
    return resample_to_output(nii, voxel_sizes=spacing_mm, order=order)


def to_mask_u8(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.uint8:
        x = (x > 0.5).astype(np.uint8)
    return x


def centroid_split_idx(mask_u8: np.ndarray) -> int:
    coords = np.argwhere(mask_u8 > 0)
    if coords.shape[0] == 0:
        return mask_u8.shape[0] // 2
    cx = float(coords[:, 0].mean())
    idx = int(round(cx))
    return max(1, min(mask_u8.shape[0] - 1, idx))


def split_flip(arr: np.ndarray, split_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    left = arr[:split_idx, :, :]
    right = arr[split_idx:, :, :]
    right_flipped = np.flip(right, axis=0).copy()
    return left, right_flipped


def save_nii_like(arr: np.ndarray, out_path: str):
    arr = np.asarray(arr)
    affine = np.eye(4, dtype=np.float32)
    nii = nib.Nifti1Image(arr, affine=affine)
    nib.save(nii, out_path)


def find_mra_and_mask_files(excel_id: str, mra_dir: str, mask_dir: str,
                            file_prefix: str = "") -> Optional[Tuple[str, str]]:
    """
    Find matching MRA and Mask files for a given patient ID.

    Args:
        excel_id: Patient ID from Excel spreadsheet
        mra_dir: Directory containing MRA files
        mask_dir: Directory containing vessel mask files
        file_prefix: Optional filename prefix (e.g. "SUB-") prepended to IDs

    Returns:
        (mra_path, mask_path) or None if not found
    """
    excel_id_str = str(excel_id)

    # Find mask file
    mask_path = None
    mask_patterns = [
        f"{file_prefix}{excel_id_str}_0000_brain.nii.gz",
    ]
    for pattern in mask_patterns:
        matches = glob.glob(os.path.join(mask_dir, pattern))
        if matches:
            mask_path = matches[0]
            break

    if mask_path is None:
        all_mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')]
        for mask_file in all_mask_files:
            file_id = mask_file.replace(file_prefix, "", 1).split("_")[0] if file_prefix else mask_file.split("_")[0]
            if excel_id_str == file_id or excel_id_str in file_id or file_id in excel_id_str:
                mask_path = os.path.join(mask_dir, mask_file)
                break

    if mask_path is None:
        return None

    # Extract patient_id from mask filename
    mask_basename = os.path.basename(mask_path)
    patient_id_from_mask = mask_basename.replace(file_prefix, "", 1).split("_")[0] if file_prefix else mask_basename.split("_")[0]

    # Find MRA file
    mra_pattern = f"{file_prefix}{patient_id_from_mask}_0000_brain_0000.nii.gz"
    mra_path = os.path.join(mra_dir, mra_pattern)

    if not os.path.exists(mra_path):
        mra_pattern = f"{file_prefix}{patient_id_from_mask}_0000_brain_0001.nii.gz"
        mra_path = os.path.join(mra_dir, mra_pattern)

    if not os.path.exists(mra_path):
        all_mra_files = [f for f in os.listdir(mra_dir) if f.endswith('.nii.gz')]
        for mra_file in all_mra_files:
            file_id = mra_file.replace(file_prefix, "", 1).split("_")[0] if file_prefix else mra_file.split("_")[0]
            if patient_id_from_mask == file_id:
                mra_path = os.path.join(mra_dir, mra_file)
                break

    if not os.path.exists(mra_path):
        return None

    return mra_path, mask_path


def main():
    parser = argparse.ArgumentParser(description="Build hemisphere dataset for Houkin grading")
    parser.add_argument("--excel_path", type=str, required=True,
                        help="Path to Houkin grade Excel file (with columns: ID, L_Grade, R_Grade)")
    parser.add_argument("--mra_dir", type=str, required=True,
                        help="Directory containing MRA images (.nii.gz)")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Directory containing vessel mask images (.nii.gz)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for hemisphere data")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Isotropic spacing in mm (default: 1.0 1.0 1.0)")
    parser.add_argument("--file_prefix", type=str, default="",
                        help="Filename prefix before patient ID (e.g. 'SUB-'). Default: no prefix")
    args = parser.parse_args()

    excel_path = args.excel_path
    mra_dir = args.mra_dir
    mask_dir = args.mask_dir
    output_dir = args.output_dir
    spacing_mm = tuple(args.spacing)
    file_prefix = args.file_prefix

    out_mra_dir = os.path.join(output_dir, "hemis_mra")
    out_mask_dir = os.path.join(output_dir, "hemis_mask")
    os.makedirs(out_mra_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "manifest_hemi.csv")

    # Read Excel
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path)
    if len(df) == 0:
        raise ValueError("Excel file is empty")

    print(f"[INFO] Read Excel: {len(df)} rows")
    print(f"[INFO] Columns: {list(df.columns)}")

    # Resume support
    done = set()
    if os.path.exists(manifest_path):
        try:
            df_done = pd.read_csv(manifest_path, encoding="utf-8-sig")
            df_done_unique = df_done.drop_duplicates(subset=['patient_id', 'hemi'], keep='first')
            for _, r in df_done_unique.iterrows():
                done.add((str(r["patient_id"]), str(r["hemi"])))
            print(f"[INFO] Resuming: {len(done)} hemisphere pairs already processed")
        except Exception as e:
            print(f"[WARN] Failed to load existing manifest: {e}")

    need_header = not os.path.exists(manifest_path)
    n_total, n_ok, n_skip, n_error = len(df), 0, 0, 0

    with open(manifest_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["patient_id", "hemi", "grade", "label3", "mra_path", "mask_path",
                         "source_mra", "source_mask"],
        )
        if need_header:
            w.writeheader()

        for i, row in enumerate(df.itertuples(index=False), start=1):
            excel_id = str(row.ID)
            gL = int(row.L_Grade) if pd.notna(row.L_Grade) and str(row.L_Grade).strip() != "" else None
            gR = int(row.R_Grade) if pd.notna(row.R_Grade) and str(row.R_Grade).strip() != "" else None

            if gL is None or gR is None:
                print(f"[WARN] Skipping {excel_id}: incomplete grade info")
                n_skip += 1
                continue

            file_match = find_mra_and_mask_files(excel_id, mra_dir, mask_dir, file_prefix)
            if file_match is None:
                print(f"[WARN] Skipping {excel_id}: MRA/Mask files not found")
                n_error += 1
                continue

            mra_path, mask_path = file_match
            patient_id = excel_id

            out_mra_L = os.path.join(out_mra_dir, f"{patient_id}_L_grade{gL}.nii.gz")
            out_mra_R = os.path.join(out_mra_dir, f"{patient_id}_R_grade{gR}.nii.gz")
            out_mask_L = os.path.join(out_mask_dir, f"{patient_id}_L_grade{gL}.nii.gz")
            out_mask_R = os.path.join(out_mask_dir, f"{patient_id}_R_grade{gR}.nii.gz")

            if (patient_id, "L") in done and (patient_id, "R") in done:
                continue

            try:
                mra = resample_iso(load_canonical(mra_path), spacing_mm, order=1)
                msk = resample_iso(load_canonical(mask_path), spacing_mm, order=0)
            except Exception as e:
                print(f"[WARN] Load/resample failed {patient_id}: {e}")
                n_error += 1
                continue

            mra_np = mra.get_fdata(dtype=np.float32)
            msk_np = to_mask_u8(msk.get_fdata(dtype=np.float32))
            split_idx = centroid_split_idx(msk_np)
            mra_L, mra_R = split_flip(mra_np, split_idx)
            msk_L, msk_R = split_flip(msk_np, split_idx)

            if (patient_id, "L") not in done:
                if not os.path.exists(out_mra_L):
                    save_nii_like(mra_L.astype(np.float32), out_mra_L)
                if not os.path.exists(out_mask_L):
                    save_nii_like(msk_L.astype(np.uint8), out_mask_L)
                w.writerow({
                    "patient_id": patient_id, "hemi": "L", "grade": gL,
                    "label3": grade_to_label3(gL), "mra_path": out_mra_L,
                    "mask_path": out_mask_L, "source_mra": mra_path, "source_mask": mask_path,
                })
                f.flush()
                done.add((patient_id, "L"))
                n_ok += 1

            if (patient_id, "R") not in done:
                if not os.path.exists(out_mra_R):
                    save_nii_like(mra_R.astype(np.float32), out_mra_R)
                if not os.path.exists(out_mask_R):
                    save_nii_like(msk_R.astype(np.uint8), out_mask_R)
                w.writerow({
                    "patient_id": patient_id, "hemi": "R", "grade": gR,
                    "label3": grade_to_label3(gR), "mra_path": out_mra_R,
                    "mask_path": out_mask_R, "source_mra": mra_path, "source_mask": mask_path,
                })
                f.flush()
                done.add((patient_id, "R"))
                n_ok += 1

            if i % 10 == 0:
                print(f"[PROGRESS] {i}/{n_total} | ok={n_ok} skip={n_skip} error={n_error}", flush=True)

    print(f"\n[DONE] manifest: {manifest_path}")
    print(f"  Processed: {n_ok} hemisphere samples")
    print(f"  Skipped: {n_skip} patients (incomplete grade)")
    print(f"  Errors: {n_error} patients (file not found / processing failed)")


if __name__ == "__main__":
    main()
