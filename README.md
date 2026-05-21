# DinoVision: Multi-Model Ensemble for Kaputt Defect Detection

This repository contains the code for reproducing our solution to the Kaputt defect detection challenge. The approach leverages pseudo-label generation, semi-supervised training with multiple DINOv2/DINOv3 backbones, multi-scale test-time augmentation (TTA), and rank-based ensemble fusion.

## Requirements

- **OS**: Linux
- **GPU**: At least 4x NVIDIA A100 80GB (8x recommended)
- **Package manager**: [uv](https://docs.astral.sh/uv/)
  
## Overview

| Stage | Description | Output |
|-------|-------------|--------|
| 1 | Environment Setup | uv virtual environment with mmpretrain |
| 2 | Pseudo-label Generation | `pseudo_labels_ref.csv` |
| 3 | Semi-supervised Training (3 models) | 3 prediction CSVs |
| 4 | Multi-head + Data Cleaning (1 model) | 1 prediction CSV |
| 5 | Ensemble Fusion | `fused_rank_avg.csv` |
| 6 | Post-processing | Final submission CSV |

## Pre-trained Weights

All pre-trained weights are available at:  
**[Google Drive](https://drive.google.com/drive/u/1/folders/1ZKx3iNjWkCZHT1tR_toe3QWYoER3iff1)**

| Weight File | Stage | Model |
|-------------|-------|-------|
| `epoch_12(gen_pseudolabel).pth` | Pseudo-label generation | DINOv3-L |
| `epoch_15(dinov3 l-v1).pth` | Semi-supervised v1 | DINOv3-L |
| `epoch_10(dinov3-h-v2).pth` | Semi-supervised v2 | DINOv3-H |
| `epoch_9(dinov2-l-v3).pth` | Semi-supervised v3 | DINOv2-L |
| `epoch_10(dinov2-l-v4).pth` | Multi-head + Clean | DINOv2-L |

---

## Project Structure

```
dinovision_code/
├── gen_pseudo-labeling/       # Stage 2: Pseudo-label generation
│   ├── configs/
│   │   └── large_728_three_freeze.py
│   ├── mmpretrain_kaputt/    # Custom MMPretrain modules
│   ├── train_vit_mm.py
│   └── generate_pseudo_labels.py
├── dinov3 l-v1/               # Stage 3: DINOv3-L semi-supervised training
│   ├── configs/
│   │   └── large_728_ref.py
│   ├── mmpretrain_kaputt/
│   ├── train_vit_mm.py
│   └── evaluate_with_ref.py
├── dinov3-h-v2/               # Stage 3: DINOv3-H semi-supervised training
│   ├── configs/
│   │   └── H_large_728_ref.py
│   ├── mmpretrain_kaputt/
│   ├── train_vit_mm.py
│   └── evaluate_with_ref.py
├── dinov2-l-v3/               # Stage 3: DINOv2-L semi-supervised training
│   ├── configs/
│   │   └── dinov2-large_728_ref.py
│   ├── mmpretrain_kaputt/
│   ├── train_vit_mm.py
│   └── evaluate_with_ref.py
├── dinov2-l-v4/               # Stage 4: Multi-head + data cleaning
│   ├── configs/
│   │   ├── large_728_mlp.py
│   │   └── large_728_mlp_clean2.py
│   ├── mmpretrain_kaputt/
│   ├── clean_mislabel.py
│   ├── train_vit_mm.py
│   ├── evaluate_with_ref.py
│   └── you_only_run_once.sh
├── post-process/              # Stage 5 & 6: Fusion and post-processing
│   ├── fusion_csv.py
│   ├── find_small_crop.py
│   └── replace_pred.py
└── README.md
```

---

## Stage 1: Environment Setup

We use [uv](https://github.com/astral-sh/uv) as the package manager for fast, reproducible environment creation.

### 1.1 Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.2 Initialize Project Environment

Each sub-folder shares the same dependency set. Use the `pyproject.toml` in any sub-directory (e.g., `dinov2-l-v4/`) as the reference:

```bash
git clone https://github.com/teamdinovision/VAND2026_Kaputt && cd VAND2026_Kaputt
uv sync
```

### 1.3 Key Dependencies

| Package | Version | Note |
|---------|---------|------|
| Python | 3.10.20 | Required |
| PyTorch | 2.1.2 (CUDA 12.1) | GPU training |
| torchvision | 0.16.2 | |
| mmpretrain | 1.2.0 | MMPretrain framework |
| mmengine | 0.10.0 | |
| mmcv | 2.1.0 | Pre-built wheel for cu121/torch2.1 |
| timm | 1.0.20 | DINOv3 backbone support |
| pandas | 2.0.0 | |
| scikit-learn | 1.3.0 | |

The `pyproject.toml` specifies:
- PyTorch from the official cu121 index
- mmcv from the OpenMMLab pre-built wheel
- Other packages from PyPI (or Aliyun mirror)

### 1.4 Data Preparation

Organize your datasets as follows. If the data are in another path, modify the path accordingly in the following script:

```
/data/public/dataset/
├── kaputt/                    # Kaputt1 dataset
│   ├── data/
│   │   ├── train/query-data/crop/
│   │   ├── validation/query-data/crop/
│   │   └── test/query-data/crop/
│   ├── *.parquet              # Label parquet files
│   └── reference_crop/crop/   # Reference images
└── kaputt2/                   # Kaputt2 dataset
    ├── data/
    │   └── test/query-data/crop/
    └── *.parquet
```

---

## Stage 2: Pseudo-Label Generation

This stage trains a DINOv3-L model on Kaputt1 query data, then uses it to assign pseudo-labels to Kaputt1 reference images (which are nominally "non-defective" but ~20% contain defects).

### 2.1 Train the Pre-training Model

```bash
cd gen_pseudo-labeling

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 \
    train_vit_mm.py configs/large_728_three_freeze.py
```

This trains a DINOv3-L backbone with gradual unfreezing, focal loss, and multi-scale augmentation. Take the **epoch 12** checkpoint.

### 2.2 Generate Pseudo-Labels for Reference Images

```bash
cd gen_pseudo-labeling

uv run python generate_pseudo_labels.py \
    --config configs/large_728_three_freeze.py \
    --checkpoint epoch_12(gen_pseudolabel).pth \
    --data-root /data/public/dataset/reference_crop/crop/ \
    --parquet-root /data/public/dataset/kaputt \
    --output pseudo_labels_ref.csv \
    --tta-scales 768 1024 1280 1408
```

**Output:** `pseudo_labels_ref.csv` — confidence-scored pseudo-labels for all reference images.

---

## Stage 3: Semi-Supervised Training (3 Models)

Using the pseudo-labels from Stage 2, train three models with different backbones. Copy `pseudo_labels_ref.csv` into each model's directory before training.

### 3.1 Model A: DINOv3-L (v1)

```bash
cd "dinov3 l-v1"

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 \
    train_vit_mm.py configs/large_728_ref.py
```

**Output weight:** `epoch_15(dinov3 l-v1).pth`

### 3.2 Model B: DINOv3-H (v2)

```bash
cd dinov3-h-v2

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 \
    train_vit_mm.py configs/H_large_728_ref.py
```

**Output weight:** `epoch_10(dinov3-h-v2).pth`

### 3.3 Model C: DINOv2-L (v3)

```bash
cd dinov2-l-v3

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 \
    train_vit_mm.py configs/dinov2-large_728_ref.py
```

**Output weight:** `epoch_9(dinov2-l-v3).pth`

### 3.4 Evaluate All Three Models on Kaputt2 Test

Each model is evaluated with multi-scale TTA and reference-image fusion.

**Model A (DINOv3-L v1):**

```bash
cd "dinov3 l-v1"

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python evaluate_with_ref.py \
    --config configs/large_728_ref.py \
    --checkpoint "epoch_15(dinov3 l-v1).pth" \
    --kaputt-root /data/public/dataset/kaputt2 \
    --split test --tta \
    --tta-scales 768 880 1024 1152 1280 1408 1600 \
    --multi-gpu \
    --anomaly-agg mean \
    --ref-scales 720 1024 \
    --output-dir dinov3-l
```

**Result:** `dinov3-l/pred_cls_anomaly_alpha0.05.csv` → rename to `prediction-dinov3l-v1.csv`

**Model B (DINOv3-H v2):**

```bash
cd dinov3-h-v2

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python evaluate_with_ref.py \
    --config configs/H_large_728_ref.py \
    --checkpoint "epoch_10(dinov3-h-v2).pth" \
    --kaputt-root /data/public/dataset/kaputt2 \
    --split test --tta \
    --tta-scales 768 880 1024 1152 1280 1408 1600 \
    --multi-gpu \
    --anomaly-agg mean \
    --ref-scales 720 1024 \
    --output-dir dinov3-h
```

**Result:** `dinov3-h/pred_cls_anomaly_alpha0.05.csv` → rename to `prediction_dinovh-v2.csv`

**Model C (DINOv2-L v3):**

```bash
cd dinov2-l-v3

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python evaluate_with_ref.py \
    --config configs/dinov2-large_728_ref.py \
    --checkpoint "epoch_9(dinov2-l-v3).pth" \
    --kaputt-root /data/public/dataset/kaputt2 \
    --split test --tta \
    --tta-scales 770 882 1022 1148 1274 1400 1596 \
    --multi-gpu \
    --anomaly-agg mean \
    --ref-scales 728 1022 \
    --output-dir dinov2-l-w
```

**Result:** `dinov2-l-w/pred_cls_anomaly_alpha0.05.csv` → rename to `predictions_dinov2l-v3.csv`

---

## Stage 4: Multi-Head Classification + Data Cleaning (DINOv2-L v4)

This stage uses a multi-classification head and iterative data cleaning to train a fourth model.

### 4.1 One-Shot Training Pipeline

The `you_only_run_once.sh` script automates the full pipeline:
1. Train initial model with multi-class head
2. Clean mislabeled data using the trained model
3. Re-train on clean data

```bash
cd dinov2-l-v4

# Edit environment variables in the script first:
# DATA_ROOT, PARQUET_ROOT, DATA2_ROOT

bash you_only_run_once.sh
```

Or run steps manually:

```bash
cd dinov2-l-v4

# Step 1: Train initial model
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 \
    train_vit_mm.py configs/large_728_mlp.py \
    --work-dir work_dirs/large_728_mlp

# Step 2: Clean mislabeled samples
uv run python clean_mislabel.py \
    --config configs/large_728_mlp.py \
    --checkpoint work_dirs/large_728_mlp/epoch_10.pth \
    --thresh-up 0.75 --thresh-down 0.3 \
    --output-dir ./data

# Step 3: Re-train on cleaned data
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 \
    train_vit_mm.py configs/large_728_mlp_clean2.py \
    --work-dir work_dirs/large_728_mlp_clean2
```

**Output weight:** `epoch_10(dinov2-l-v4).pth` (from `work_dirs/large_728_mlp_clean2/epoch_10.pth`)

### 4.2 Evaluate on Kaputt2 Test

```bash
cd dinov2-l-v4

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python evaluate_with_ref.py \
    --config configs/large_728_mlp_clean2.py \
    --checkpoint "epoch_10(dinov2-l-v4).pth" \
    --kaputt-root /data/public/dataset/kaputt2 \
    --split test --tta \
    --tta-scales 756 868 1022 1148 1274 1400 1596 \
    --multi-gpu \
    --anomaly-agg mean \
    --ref-scales 720 1024 \
    --output-dir dinov2-l-m
```

**Result:** `dinov2-l-m/pred_cls_anomaly_alpha0.05.csv` → rename to `predictions_dinov2l-v4.csv`

---

## Stage 5: Ensemble Fusion

Fuse predictions from all 4 models using rank-based averaging.

### 5.1 Prepare Input CSVs

Place the following files in the `post-process/` directory:
- `prediction-dinov3l-v1.csv` (from Stage 3.4, Model A)
- `prediction_dinoh-v2.csv` (from Stage 3.4, Model B)
- `predictions div2l-v3.csv` (from Stage 3.4, Model C)
- `predictions div2l-v4.csv` (from Stage 4.2, Model D)

> **Note:** The fusion script (`fusion_csv.py`) reads these exact filenames. Ensure file names match.

### 5.2 Run Fusion

```bash
cd post-process

uv run python fusion_csv.py
```

**Output:** `fused_outputs/fused_rank_avg.csv` — rank-averaged ensemble predictions.

Copy it to the working directory:
```bash
cp fused_outputs/fused_rank_avg.csv ./fused_rank_avg.csv
```

---

## Stage 6: Post-Processing

We observe that images smaller than 200×200 pixels are typically non-defective crops. We replace their prediction scores with a low value (0.01).

### 6.1 Find Small Images

```bash
cd post-process

uv run python find_small_crop.py
```

This scans the Kaputt2 test crop directory and outputs `small_image_k2.csv` listing all images with width < 200px.

### 6.2 Replace Predictions for Small Images

```bash
cd post-process

uv run python replace_pred.py
```

**Final Output:** `prediction_dinovision.csv` — this is the final submission file.

---

## Quick Reproduction (Using Pre-trained Weights)

If you only want to reproduce the final result without full training, download all weights from [Google Drive](https://drive.google.com/drive/u/1/folders/1ZKx3iNjWkCZHT1tR_toe3QWYoER3iff1) and skip training steps:

```bash
# 1. Setup environment
cd dinov2-l-v4 && uv sync && cd ..

# 2. Evaluate Model A
cd "dinov3 l-v1"
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python evaluate_with_ref.py \
    --config configs/large_728_ref.py \
    --checkpoint "epoch_15(dinov3 l-v1).pth" \
    --kaputt-root /data/public/dataset/kaputt2 \
    --split test --tta \
    --tta-scales 768 880 1024 1152 1280 1408 1600 \
    --multi-gpu --anomaly-agg mean --ref-scales 720 1024 \
    --output-dir dinov3-l
cd ..

# 3. Evaluate Model B
cd dinov3-h-v2
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python evaluate_with_ref.py \
    --config configs/H_large_728_ref.py \
    --checkpoint "epoch_10(dinov3-h-v2).pth" \
    --kaputt-root /data/public/dataset/kaputt2 \
    --split test --tta \
    --tta-scales 768 880 1024 1152 1280 1408 1600 \
    --multi-gpu --anomaly-agg mean --ref-scales 720 1024 \
    --output-dir dinov3-h
cd ..

# 4. Evaluate Model C
cd dinov2-l-v3
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python evaluate_with_ref.py \
    --config configs/dinov2-large_728_ref.py \
    --checkpoint "epoch_9(dinov2-l-v3).pth" \
    --kaputt-root /data/public/dataset/kaputt2 \
    --split test --tta \
    --tta-scales 770 882 1022 1148 1274 1400 1596 \
    --multi-gpu --anomaly-agg mean --ref-scales 728 1022 \
    --output-dir dinov2-l-w
cd ..

# 5. Evaluate Model D
cd dinov2-l-v4
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python evaluate_with_ref.py \
    --config configs/large_728_mlp_clean2.py \
    --checkpoint "epoch_10(dinov2-l-v4).pth" \
    --kaputt-root /data/public/dataset/kaputt2 \
    --split test --tta \
    --tta-scales 756 868 1022 1148 1274 1400 1596 \
    --multi-gpu --anomaly-agg mean --ref-scales 720 1024 \
    --output-dir dinov2-l-m
cd ..

# 6. Collect prediction CSVs into post-process/
cp "dinov3 l-v1/dinov3-l/pred_cls_anomaly_alpha0.05.csv" post-process/prediction-dinov3l-v1.csv
cp "dinov3-h-v2/dinov3-h/pred_cls_anomaly_alpha0.05.csv" post-process/prediction_dinoh-v2.csv
cp "dinov2-l-v3/dinov2-l-w/pred_cls_anomaly_alpha0.05.csv" post-process/predictions_div2l-v3.csv
cp "dinov2-l-v4/dinov2-l-m/pred_cls_anomaly_alpha0.05.csv" post-process/predictions_dinov2l-v4.csv

# 7. Ensemble fusion
cd post-process
uv run python fusion_csv.py
cp fused_outputs/fused_rank_avg.csv ./fused_rank_avg.csv

# 8. Post-processing
uv run python find_small_crop.py
uv run python replace_pred.py
cd ..
```

The final submission file is `post-process/prediction_dinovision.csv`.

---

## Hardware Requirements

- **GPU:** 4× NVIDIA A100 (80GB) or equivalent (for multi-GPU training)
- **RAM:** 64GB+ recommended
- **Storage:** ~50GB for weights + datasets

## Notes

- All training uses mixed precision (AMP) for memory efficiency.
- The DINOv3-H model (`dinov3-h-v2`) requires significantly more GPU memory due to the huge backbone.
- TTA scales are tuned per-model to align with respective patch sizes (patch=16 for DINOv3, patch=14 for DINOv2).
- The rank-based fusion is chosen because AP (Average Precision) is a ranking metric — only the ordering of predictions matters, not absolute probability values.
