# VMG-MoyaNet

**Vessel Mask-Guided Dual-Channel Deep Learning for Differential Diagnosis and Houkin Grading of Moyamoya Disease on MRA**

A multi-center validation study for automated diagnosis and grading of Moyamoya Disease (MMD) using MRA images.

## Overview

VMG-MoyaNet is a deep learning framework for:

1. **Differential Diagnosis**: Distinguishing MMD from Intracranial Atherosclerotic Stenosis (ICAS) and Normal Controls (NC)
2. **Houkin Grading**: Classifying the severity of Moyamoya Disease (Grade 1/2/3-4)

### Key Features

- **Dual-channel input**: MRA image + Vessel Mask (from COSTA segmentation)
- **Vessel mask guidance**: Leverages cerebrovascular structure for improved accuracy
- **Multi-center validation**: Validated across multiple clinical centers
- **End-to-end pipeline**: From raw MRA to diagnosis/grading

## Architecture

```
                         +---------------------+
                         |    Raw MRA Image     |
                         +----------+----------+
                                    |
                    +---------------+---------------+
                    v                               v
         +------------------+            +------------------+
         |  BET2 Skull Strip |            |    COSTA Model   |
         +--------+---------+            | (Vessel Segment) |
                  |                       +--------+---------+
                  |                                |
                  +---------------+---------------+
                                  v
                    +-------------------------+
                    |  Dual-Channel Input      |
                    |  [MRA | Vessel Mask]     |
                    +------------+------------+
                                 |
              +------------------+------------------+
              v                                      v
    +---------------------+              +---------------------+
    |   Houkin Grading     |              | Differential Diag.  |
    |   (Hemisphere-level) |              |   (Full-brain)      |
    |   DenseNet-121 3D    |              |   DenseNet-121 3D   |
    |   160x160x160        |              |   224x224x224       |
    +---------------------+              +---------------------+
              |                                      |
              v                                      v
    +---------------------+              +---------------------+
    |  G1 / G2 / G3-4     |              |  MMD / ICAS / NC    |
    +---------------------+              +---------------------+
```

## Installation

### Prerequisites

- Python >= 3.8
- CUDA >= 11.0 (for GPU training)
- FSL (for BET2 skull stripping) - [Installation Guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation)
- COSTA (for vessel segmentation) - [GitHub](https://github.com/iMED-Lab/COSTA)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/HanGhost001/VMG-MoyaNet.git
cd VMG-MoyaNet

# Create virtual environment
conda create -n vmg python=3.9
conda activate vmg

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install COSTA for vessel segmentation (required for preprocessing)
pip install nnunet
# Follow COSTA setup: https://github.com/iMED-Lab/COSTA
```

## Usage

### 1. Preprocess MRA Images (Vessel Segmentation)

The preprocessing pipeline uses COSTA for vessel segmentation. COSTA must be installed separately.

```bash
# Complete preprocessing: skull stripping + histogram standardization + COSTA segmentation
python vessel_segmentation/segment_vessels.py \
    -i /path/to/raw_mra \
    -o /path/to/preprocessed \
    -m /path/to/costa_model
```

### 2. Build Hemisphere Dataset (for Houkin Grading)

```bash
python -m scripts.build_hemi_dataset \
    --excel_path /path/to/houkin_grade.xlsx \
    --mra_dir /path/to/mra \
    --mask_dir /path/to/masks \
    --output_dir /path/to/hemi_data
```

### 3. Split Dataset

```bash
python -m scripts.split_dataset \
    --manifest /path/to/manifest.csv \
    --output_dir /path/to/splits \
    --seed 42
```

### 4. Train Models

Both training scripts are driven by YAML configuration files.

**Houkin Grading:**
```bash
python -m houkin_grading.train_hemi_fusion --config configs/houkin_grading.yaml
```

**Differential Diagnosis:**
```bash
python -m differential_diagnosis.train_full_fusion --config configs/differential_diagnosis.yaml
```

Before training, update the `paths` section in the YAML config to point to your data:
```yaml
paths:
  train_csv: "data/splits/train.csv"
  val_csv: "data/splits/val.csv"
  test_csv: "data/splits/test.csv"
  output_dir: "runs/houkin_grading"
```

### 5. Evaluate

```bash
# Houkin grading evaluation
python -m houkin_grading.eval_hemi_fusion --run_dir /path/to/run

# Differential diagnosis ensemble evaluation
python -m differential_diagnosis.eval_ensemble_full /path/to/run1 /path/to/run2 --test_csv /path/to/test.csv
```

## Project Structure

```
VMG-MoyaNet/
+-- README.md
+-- LICENSE
+-- requirements.txt
+-- configs/
|   +-- houkin_grading.yaml
|   +-- differential_diagnosis.yaml
+-- vessel_segmentation/
|   +-- segment_vessels.py           # Preprocessing pipeline entry
|   +-- costa/
|       +-- network_architecture/
|       |   +-- costa.py             # COSTA network (reference)
|       |   +-- swin_transformer.py  # Swin Transformer (reference)
|       +-- preprocessing/
|       |   +-- skull_stripping.py
|       |   +-- intensity_hist_stand.py
|       |   +-- landmarks.pth
|       +-- inference/
|           +-- predict.py           # COSTA inference (reference)
+-- houkin_grading/
|   +-- model_3d_densenet.py         # DenseNet-121 3D
|   +-- model_3d_fusion.py           # ResNet 3D (ablation)
|   +-- dataset_hemi_fusion.py
|   +-- train_hemi_fusion.py
|   +-- eval_hemi_fusion.py
|   +-- gradcam_utils.py
|   +-- utils_metrics.py
+-- differential_diagnosis/
|   +-- model_3d_densenet.py         # DenseNet-121 3D
|   +-- model_3d_fusion.py           # ResNet 3D (ablation)
|   +-- dataset_full_fusion.py
|   +-- train_full_fusion.py
|   +-- eval_ensemble_full.py
|   +-- gradcam_utils.py
|   +-- utils_metrics.py
+-- scripts/
    +-- preprocess_mra.py
    +-- build_hemi_dataset.py
    +-- split_dataset.py
```

## Configuration

Training is configured via YAML files in `configs/`. Key parameters:

- `model.architecture`: `"densenet121"` (default) or ResNet variants for ablation
- `model.growth_rate`: DenseNet growth rate (default: 56, ~30M params)
- `data.in_channels`: 2 (MRA + Vessel Mask)
- `data.use_nonzero_normalization`: Non-zero voxel normalization
- `training.loss.type`: `"focal"` with per-class alpha weights
- `paths.*`: Data paths (must be updated before training)

See `configs/houkin_grading.yaml` and `configs/differential_diagnosis.yaml` for full configuration options.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{vmg-moyanet2026,
  title={Vessel mask-guided dual-channel deep learning for differential diagnosis
         and Houkin grading of moyamoya disease on MRA: a multicenter validation study},
  author={[Authors]},
  journal={[Journal]},
  year={2026}
}
```

## Acknowledgements

- [COSTA](https://github.com/iMED-Lab/COSTA) - Cerebral artery segmentation
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet) - Medical image segmentation framework

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Medical Disclaimer

This software is intended for research purposes only and is NOT approved for clinical use. The predictions made by this model should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions regarding medical conditions.

## Contact

For questions or collaborations, please open an issue or contact the authors.
