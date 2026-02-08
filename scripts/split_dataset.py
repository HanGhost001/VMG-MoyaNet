"""
Dataset Splitting Script

This script splits the dataset into train/validation/test sets
with patient-level stratification to avoid data leakage.

For Houkin Grading:
- Split by patient ID (not hemisphere)
- Stratify by maximum grade per patient
- Default ratio: 70% train, 15% val, 15% test

For Differential Diagnosis:
- Split by patient ID
- Stratify by diagnosis class (MMD/ICAS/NC)
- Default ratio: 70% train, 15% val, 15% test

Usage:
    python split_dataset.py --manifest <manifest.csv> --output_dir <output_folder>
"""

import os
import argparse
import random
from collections import defaultdict
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np


def patient_level_split(
    manifest_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_col: str = "label3",
    patient_col: str = "patient_id",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset at patient level with stratification.
    
    Args:
        manifest_df: DataFrame with patient data
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        stratify_col: Column to stratify by
        patient_col: Column containing patient IDs
        seed: Random seed
    
    Returns:
        (train_df, val_df, test_df)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Group samples by patient
    patient_groups = defaultdict(list)
    for idx, row in manifest_df.iterrows():
        patient_groups[row[patient_col]].append(idx)
    
    # Get the stratification label for each patient (use max if multiple samples)
    patient_labels = {}
    for patient_id, indices in patient_groups.items():
        labels = manifest_df.loc[indices, stratify_col].values
        patient_labels[patient_id] = int(max(labels))  # Use max grade for stratification
    
    # Group patients by their label
    label_patients = defaultdict(list)
    for patient_id, label in patient_labels.items():
        label_patients[label].append(patient_id)
    
    train_patients = []
    val_patients = []
    test_patients = []
    
    # Split each label group proportionally
    for label, patients in label_patients.items():
        random.shuffle(patients)
        n = len(patients)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_patients.extend(patients[:n_train])
        val_patients.extend(patients[n_train:n_train + n_val])
        test_patients.extend(patients[n_train + n_val:])
    
    # Convert patient lists to sample indices
    train_indices = []
    val_indices = []
    test_indices = []
    
    for patient_id in train_patients:
        train_indices.extend(patient_groups[patient_id])
    for patient_id in val_patients:
        val_indices.extend(patient_groups[patient_id])
    for patient_id in test_patients:
        test_indices.extend(patient_groups[patient_id])
    
    train_df = manifest_df.loc[train_indices].reset_index(drop=True)
    val_df = manifest_df.loc[val_indices].reset_index(drop=True)
    test_df = manifest_df.loc[test_indices].reset_index(drop=True)
    
    return train_df, val_df, test_df


def print_split_stats(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                      label_col: str = "label3"):
    """Print statistics about the split."""
    print("\n" + "=" * 50)
    print("Split Statistics")
    print("=" * 50)
    
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{name}: {len(df)} samples")
        if label_col in df.columns:
            label_counts = df[label_col].value_counts().sort_index()
            for label, count in label_counts.items():
                print(f"  Label {label}: {count} ({100*count/len(df):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Split dataset for VMG-MoyaNet")
    parser.add_argument(
        "-m", "--manifest",
        required=True,
        help="Path to manifest CSV file"
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Directory to save split CSV files"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--stratify_col",
        default="label3",
        help="Column to stratify by (default: label3)"
    )
    parser.add_argument(
        "--patient_col",
        default="patient_id",
        help="Patient ID column (default: patient_id)"
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    manifest_df = pd.read_csv(args.manifest)
    print(f"Total samples: {len(manifest_df)}")
    
    # Split
    train_df, val_df, test_df = patient_level_split(
        manifest_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify_col=args.stratify_col,
        patient_col=args.patient_col,
        seed=args.seed
    )
    
    # Print statistics
    print_split_stats(train_df, val_df, test_df, args.stratify_col)
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.csv")
    val_path = os.path.join(args.output_dir, "val.csv")
    test_path = os.path.join(args.output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved splits to: {args.output_dir}")
    print(f"  - train.csv: {len(train_df)} samples")
    print(f"  - val.csv: {len(val_df)} samples")
    print(f"  - test.csv: {len(test_df)} samples")


if __name__ == "__main__":
    main()
