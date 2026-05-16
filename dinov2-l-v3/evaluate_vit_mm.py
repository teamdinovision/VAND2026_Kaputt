"""
Kaputt Defect Detection — MMPretrain Evaluation
================================================
Standalone evaluation script.

Supports:
  1. Standard evaluation with full metrics (AP, AUROC, R@50%P, R@1%FPR)
  2. Test-Time Augmentation (multi-scale + flip)
  3. Model Ensemble — same architecture (multiple checkpoints, one config)
  4. Multi-Architecture Ensemble — DINOv2 + ConvNeXt (multiple configs)
  5. Label-free prediction mode (--predict) — no labels required

Usage:
    # Standard evaluation (requires labels)
    python evaluate_vit_mm.py --config configs/kaputt_dinov2.py \\
        --checkpoint work_dirs/best.pth --split test

    # TTA evaluation (multi-scale + flip)
    python evaluate_vit_mm.py --config configs/kaputt_dinov2.py \\
        --checkpoint work_dirs/best.pth --tta

    # Same-architecture ensemble (multiple checkpoints)
    python evaluate_vit_mm.py --config configs/kaputt_dinov2.py \\
        --checkpoint ckpt1.pth ckpt2.pth ckpt3.pth --ensemble

    # Multi-architecture ensemble: DINOv2 + ConvNeXt
    python evaluate_vit_mm.py \\
        --config configs/kaputt_dinov2.py configs/kaputt_convnext.py \\
        --checkpoint dinov2_best.pth convnext_best.pth \\
        --ensemble --tta

    # Include SWA checkpoint in ensemble
    python evaluate_vit_mm.py \\
        --config configs/kaputt_dinov2.py configs/kaputt_dinov2.py \\
               configs/kaputt_convnext.py \\
        --checkpoint dinov2_best.pth dinov2_swa.pth convnext_best.pth \\
        --ensemble --tta

    # --- Label-free prediction (no ground truth needed) ---
    # From parquet (uses capture_id to find images, ignores defect column)
    python evaluate_vit_mm.py --config configs/kaputt_dinov2.py \\
        --checkpoint best.pth --predict --split test

    # From an image folder directly (no parquet needed)
    python evaluate_vit_mm.py --config configs/kaputt_dinov2.py \\
        --checkpoint best.pth --predict --image-dir /path/to/images/

    # Prediction + TTA + Ensemble
    python evaluate_vit_mm.py \\
        --config configs/kaputt_dinov2.py configs/kaputt_convnext.py \\
        --checkpoint dinov2.pth convnext.pth \\
        --predict --image-dir /path/to/images/ --ensemble --tta
        
    # 同架构 ensemble（1 个 config + 多个 checkpoint）
    python evaluate_vit_mm.py \
        --config configs/kaputt_dinov2.py \
        --checkpoint ckpt1.pth ckpt2.pth ckpt3.pth \
        --image-dir /path/to/images/ \
        --ensemble --tta \
        --tta-scales 512 752 1008 1248

    # 多架构 ensemble（多个 config 1:1 对应多个 checkpoint）
    python evaluate_vit_mm.py \
        --config configs/kaputt_dinov2.py configs/kaputt_convnext.py \
        --checkpoint dinov3_best.pth convnext_best.pth \
        --image-dir /path/to/images/ \
        --ensemble --tta \
        --tta-scales 512 752 1008 1248

    # --- Multi-GPU TTA (distribute scale/flip views across GPUs) ---
    # Single model, TTA on multiple GPUs
    python evaluate_vit_mm.py --config configs/kaputt_dinov2.py \
        --checkpoint best.pth --tta --multi-gpu

    # Ensemble + TTA on multiple GPUs
    python evaluate_vit_mm.py \
        --config configs/kaputt_dinov2.py configs/kaputt_convnext.py \
        --checkpoint dinov2_best.pth convnext_best.pth \
        --ensemble --tta --multi-gpu \
        --tta-scales 512 752 1008 1248

    # Control visible GPUs via environment variable
    # CUDA_VISIBLE_DEVICES=0,1 python evaluate_vit_mm.py ... --tta --multi-gpu

"""

import os
import sys
import json
import time
import argparse
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmpretrain_kaputt  # noqa: F401

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.registry import MODELS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Simple dataset (independent of mmengine for flexible TTA / ensemble)
# ---------------------------------------------------------------------------
class SimpleKaputtDataset(Dataset):
    """Lightweight dataset that keeps the DataFrame in memory.

    Supports both labeled (evaluation) and unlabeled (prediction) modes.
    When ``defect`` column is missing from df, labels default to -1.
    """

    def __init__(self, df, image_root, transform=None):
        self.df = df
        self.image_root = Path(image_root)
        self.transform = transform
        if 'defect' in df.columns:
            self.labels = df['defect'].astype(int).tolist()
        else:
            self.labels = [-1] * len(df)
        self.capture_ids = df['capture_id'].tolist()
        self.has_labels = 'defect' in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(
            self.image_root / f'{row.capture_id}.jpg').convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


class ImageFolderDataset(Dataset):
    """Dataset that reads all .jpg/.png images from a folder (no parquet).

    Used for pure prediction when only an image directory is available.
    Returns dummy label=-1 for compatibility with inference functions.
    """

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(self, image_root, transform=None):
        self.image_root = Path(image_root)
        self.transform = transform
        self.image_paths = sorted([
            p for p in self.image_root.iterdir()
            if p.suffix.lower() in self.EXTENSIONS
        ])
        self.capture_ids = [p.stem for p in self.image_paths]
        self.labels = [-1] * len(self.image_paths)
        self.has_labels = False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(-1, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def build_model(config_path, checkpoint_path, device):
    """Build mmpretrain model from config and load checkpoint weights."""
    cfg = Config.fromfile(config_path)
    cfg.model.backbone.pretrained = False

    # Allow variable input sizes for TTA (ViT needs this flag)
    model_name = cfg.model.backbone.get('model_name', '')
    if 'dinov2' in model_name or 'dinov3' in model_name:
        cfg.model.backbone.dynamic_img_size = True

    model = MODELS.build(cfg.model)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    img_size = cfg.get('img_size', 518)
    patch_size = cfg.get('patch_size', 14)
    return model, cfg, img_size, patch_size


def get_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_flip_transform(img_size, direction='horizontal'):
    ops = [
        transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    if direction == 'horizontal':
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    elif direction == 'vertical':
        ops.append(transforms.RandomVerticalFlip(p=1.0))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


# ---------------------------------------------------------------------------
# Inference routines
# ---------------------------------------------------------------------------
@torch.no_grad()
def _inference_pass(model, loader, device, amp_dtype=torch.bfloat16,
                    quiet=False):
    """Single-pass inference returning P(defective) array."""
    probs_list = []
    for images, _ in tqdm(loader, desc='  infer', leave=False,
                          file=sys.stdout, disable=quiet):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
            logits = model(images, mode='tensor')
        probs = F.softmax(logits.float(), dim=1)[:, 1]
        probs_list.append(probs.cpu())
    return torch.cat(probs_list).numpy()


def _make_dataset(df_or_dir, image_root, transform):
    """Create dataset from either a DataFrame+image_root or image directory."""
    if isinstance(df_or_dir, pd.DataFrame):
        return SimpleKaputtDataset(df_or_dir, image_root, transform)
    return ImageFolderDataset(df_or_dir, transform)


def standard_inference(model, df_or_dir, image_root, img_size, device,
                       batch_size=16, num_workers=8):
    """Standard single-scale inference."""
    ds = _make_dataset(df_or_dir, image_root, get_eval_transform(img_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    probs = _inference_pass(model, loader, device)
    labels = np.array(ds.labels)
    capture_ids = ds.capture_ids
    return labels, probs, capture_ids


def tta_inference(model, df_or_dir, image_root, base_size, patch_size, device,
                  batch_size=16, num_workers=8,
                  extra_scales=(0, 2, 4), use_flip=True,
                  explicit_scales=None):
    """Multi-scale + flip TTA.

    When ``explicit_scales`` is provided (list of pixel sizes), those exact
    sizes are used directly — every value should be a multiple of
    ``patch_size`` for ViT models.

    Otherwise falls back to the legacy patch-offset mode where
    ``extra_scales`` lists offsets in *number of patches* from the base.
    """
    if explicit_scales:
        scales = sorted(set(explicit_scales))
    elif patch_size < 2:
        scales = sorted({base_size + d * 32 for d in extra_scales})
    else:
        base_patches = base_size // patch_size
        scales = sorted({(base_patches + d) * patch_size for d in extra_scales})

    all_probs = []
    labels = None
    capture_ids = None

    for s in scales:
        print(f'  TTA scale {s}×{s}', flush=True)

        ds = _make_dataset(df_or_dir, image_root, get_eval_transform(s))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        all_probs.append(_inference_pass(model, loader, device))
        if labels is None:
            labels = np.array(ds.labels)
            capture_ids = ds.capture_ids

        if use_flip:
            ds_h = _make_dataset(
                df_or_dir, image_root, get_flip_transform(s, 'horizontal'))
            ld_h = DataLoader(ds_h, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
            all_probs.append(_inference_pass(model, ld_h, device))

            ds_v = _make_dataset(
                df_or_dir, image_root, get_flip_transform(s, 'vertical'))
            ld_v = DataLoader(ds_v, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
            all_probs.append(_inference_pass(model, ld_v, device))

    print(f'  TTA views: {len(all_probs)}', flush=True)
    avg = np.mean(all_probs, axis=0)
    return labels, avg, capture_ids


def tta_inference_multigpu(config_path, checkpoint_path, df_or_dir, image_root,
                           base_size, patch_size, batch_size=16, num_workers=8,
                           extra_scales=(0, 2, 4), use_flip=True,
                           explicit_scales=None):
    """Multi-GPU TTA: distribute scale/flip views across all available GPUs.

    Each GPU gets its own model replica and processes a subset of TTA views
    (scale x flip combinations) in parallel via threads.  Falls back to
    single-GPU ``tta_inference`` when only one GPU is visible.
    """
    from concurrent.futures import ThreadPoolExecutor

    if explicit_scales:
        scales = sorted(set(explicit_scales))
    elif patch_size < 2:
        scales = sorted({base_size + d * 32 for d in extra_scales})
    else:
        base_patches = base_size // patch_size
        scales = sorted({(base_patches + d) * patch_size
                         for d in extra_scales})

    views = []
    for s in scales:
        views.append((s, None))
        if use_flip:
            views.append((s, 'horizontal'))
            views.append((s, 'vertical'))

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        device = torch.device('cuda:0' if torch.cuda.is_available()
                              else 'cpu')
        model, _, _, _ = build_model(config_path, checkpoint_path, device)
        return tta_inference(
            model, df_or_dir, image_root, base_size, patch_size, device,
            batch_size=batch_size, num_workers=num_workers,
            extra_scales=extra_scales, use_flip=use_flip,
            explicit_scales=explicit_scales)

    workers_per_gpu = max(1, num_workers // num_gpus)
    print(f'  Multi-GPU TTA: {len(views)} views on {num_gpus} GPUs '
          f'({workers_per_gpu} dataloader workers/GPU)', flush=True)

    models = []
    for gpu_id in range(num_gpus):
        device = torch.device(f'cuda:{gpu_id}')
        m, _, _, _ = build_model(config_path, checkpoint_path, device)
        models.append((m, device))

    ds0 = _make_dataset(df_or_dir, image_root, get_eval_transform(scales[0]))
    labels = np.array(ds0.labels)
    capture_ids = ds0.capture_ids
    del ds0

    gpu_tasks = [[] for _ in range(num_gpus)]
    for i, view in enumerate(views):
        gpu_tasks[i % num_gpus].append((i, view))

    all_probs = [None] * len(views)
    total_views = len(views)
    done_counter = [0]
    counter_lock = threading.Lock()

    def _gpu_worker(gpu_id):
        model, device = models[gpu_id]
        for view_idx, (scale, flip_dir) in gpu_tasks[gpu_id]:
            tag = f'{scale}\u00d7{scale}'
            if flip_dir:
                tag += f'+{flip_dir}'
            print(f'  [GPU {gpu_id}] TTA {tag} ...', flush=True)
            t0 = time.perf_counter()
            tfm = (get_eval_transform(scale) if flip_dir is None
                   else get_flip_transform(scale, flip_dir))
            ds = _make_dataset(df_or_dir, image_root, tfm)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=workers_per_gpu, pin_memory=True)
            all_probs[view_idx] = _inference_pass(
                model, loader, device, quiet=True)
            elapsed = time.perf_counter() - t0
            with counter_lock:
                done_counter[0] += 1
                print(f'  [GPU {gpu_id}] TTA {tag} done  '
                      f'({elapsed:.1f}s)  '
                      f'[{done_counter[0]}/{total_views} views]',
                      flush=True)

    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futs = [pool.submit(_gpu_worker, gid) for gid in range(num_gpus)]
        for f in futs:
            f.result()

    del models
    torch.cuda.empty_cache()

    print(f'  TTA views: {len(views)}', flush=True)
    return labels, np.mean(all_probs, axis=0), capture_ids


# ---------------------------------------------------------------------------
# Multi-architecture ensemble
# ---------------------------------------------------------------------------
def multi_arch_ensemble(config_paths, checkpoint_paths, df_or_dir, image_root,
                        device, use_tta=False, batch_size=16, num_workers=8,
                        tta_extra_scales=(0, 2, 4), tta_explicit_scales=None,
                        multigpu=False):
    """Ensemble across different architectures (e.g. DINOv2 + ConvNeXt).

    Each (config, checkpoint) pair defines a model. All predictions are
    averaged with equal weight.

    Args:
        config_paths (list[str]): Config file for each model.
        checkpoint_paths (list[str]): Checkpoint file for each model.
        df_or_dir: DataFrame or image directory path.
        tta_extra_scales: Legacy patch-offset scales for TTA.
        tta_explicit_scales: Explicit pixel sizes for TTA (overrides offsets).
        multigpu: Distribute TTA views across multiple GPUs.
    """
    all_probs = []
    labels = None
    capture_ids = None

    use_multigpu = multigpu and torch.cuda.device_count() > 1

    for i, (cfg_path, ckpt_path) in enumerate(
            zip(config_paths, checkpoint_paths)):
        print(f'\n[Ensemble {i+1}/{len(checkpoint_paths)}] '
              f'config={cfg_path}  ckpt={ckpt_path}', flush=True)

        cfg = Config.fromfile(cfg_path)
        img_size = cfg.get('img_size', 518)
        patch_size = cfg.get('patch_size', 14)

        model_explicit = tta_explicit_scales
        if model_explicit is None:
            model_explicit = cfg.get('tta_scales', None)

        if use_tta and use_multigpu:
            lbl, probs, cids = tta_inference_multigpu(
                cfg_path, ckpt_path, df_or_dir, image_root,
                img_size, patch_size,
                batch_size=batch_size, num_workers=num_workers,
                extra_scales=tta_extra_scales,
                explicit_scales=model_explicit)
        else:
            model, _, img_size, patch_size = build_model(
                cfg_path, ckpt_path, device)
            if use_tta:
                lbl, probs, cids = tta_inference(
                    model, df_or_dir, image_root, img_size, patch_size,
                    device, batch_size=batch_size, num_workers=num_workers,
                    extra_scales=tta_extra_scales,
                    explicit_scales=model_explicit)
            else:
                lbl, probs, cids = standard_inference(
                    model, df_or_dir, image_root, img_size, device,
                    batch_size=batch_size, num_workers=num_workers)
            del model
            torch.cuda.empty_cache()

        all_probs.append(probs)
        if labels is None:
            labels = lbl
            capture_ids = cids

    avg = np.mean(all_probs, axis=0)
    return labels, avg, capture_ids


def ensemble_inference(config_path, checkpoint_paths, df_or_dir, image_root,
                       device, use_tta=False, batch_size=16, num_workers=8,
                       tta_extra_scales=(0, 2, 4), tta_explicit_scales=None,
                       multigpu=False):
    """Same-architecture ensemble: one config, multiple checkpoints."""
    config_paths = [config_path] * len(checkpoint_paths)
    return multi_arch_ensemble(
        config_paths, checkpoint_paths, df_or_dir, image_root,
        device, use_tta, batch_size, num_workers,
        tta_extra_scales=tta_extra_scales,
        tta_explicit_scales=tta_explicit_scales,
        multigpu=multigpu)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(labels, probs):
    from sklearn.metrics import (
        average_precision_score, roc_auc_score,
        precision_recall_curve, roc_curve,
        classification_report, confusion_matrix,
    )

    ap    = average_precision_score(labels, probs)
    auroc = roc_auc_score(labels, probs)

    prec, rec, _ = precision_recall_curve(labels, probs)
    r50p = rec[prec >= 0.50].max() if (prec >= 0.50).any() else 0.0

    fpr, tpr, _ = roc_curve(labels, probs)
    r1fpr = tpr[fpr <= 0.01].max() if (fpr <= 0.01).any() else 0.0

    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = np.argmax(f1)
    thresholds = precision_recall_curve(labels, probs)[2]
    thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    preds = (probs >= thresh).astype(int)
    cm = confusion_matrix(labels, preds)
    report = classification_report(
        labels, preds, target_names=['Non-defective', 'Defective'])

    return {
        'AP_any (%)':             round(ap    * 100, 2),
        'AUROC (%)':              round(auroc * 100, 2),
        'Recall@50%Precision (%)': round(r50p  * 100, 2),
        'Recall@1%FPR (%)':       round(r1fpr * 100, 2),
        'optimal_threshold':      round(thresh, 4),
        'confusion_matrix':       cm.tolist(),
        'classification_report':  report,
    }


def evaluate_by_material(df, probs, labels):
    from sklearn.metrics import average_precision_score as ap_fn
    results = {}
    if 'item_material' not in df.columns:
        return results
    for mat in sorted(df['item_material'].unique()):
        mask = df['item_material'].values == mat
        if mask.sum() < 2 or labels[mask].sum() == 0:
            continue
        results[mat] = dict(
            AP=round(ap_fn(labels[mask], probs[mask]) * 100, 2),
            count=int(mask.sum()),
            defective=int(labels[mask].sum()),
        )
    return results


def evaluate_major_only(df, probs):
    from sklearn.metrics import average_precision_score as ap_fn
    if 'major_defect' not in df.columns:
        return None
    non_def = ~df['defect'].values
    major   = df['major_defect'].values
    mask    = non_def | major
    if mask.sum() < 2 or major.sum() == 0:
        return None
    sub_labels = df['defect'].values[mask].astype(int)
    return dict(
        AP_major=round(ap_fn(sub_labels, probs[mask]) * 100, 2),
        total=int(mask.sum()),
        major_defective=int(major.sum()),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Kaputt MMPretrain — Evaluation / TTA / Ensemble')
    p.add_argument('--config', nargs='+', required=True,
                   help='Config file(s). One config = same-arch ensemble. '
                        'Multiple configs = multi-arch ensemble '
                        '(paired 1:1 with checkpoints).')
    p.add_argument('--checkpoint', nargs='+', required=True,
                   help='One or more checkpoint paths')
    p.add_argument('--data-root', default=None,
                   help='Override data root from config')
    p.add_argument('--parquet-root', default=None,
                   help='Override parquet root from config')
    p.add_argument('--split', default='test',
                   choices=['train', 'validation', 'test'])
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--output-dir', default='results_vit_mm')

    p.add_argument('--tta', action='store_true',
                   help='Enable multi-scale + flip TTA')
    p.add_argument('--tta-scales', nargs='+', type=int, default=None,
                   help='Explicit TTA scales in pixels (e.g. 512 752 1008). '
                        'Overrides --tta-extra-scales and config tta_scales. '
                        'Values should be multiples of patch_size.')
    p.add_argument('--tta-extra-scales', nargs='+', type=int,
                   default=[0, 2, 4],
                   help='Legacy: patch-count offsets from base size '
                        '(ignored when --tta-scales or config tta_scales is set)')

    p.add_argument('--ensemble', action='store_true',
                   help='Ensemble predictions across all checkpoints')
    p.add_argument('--multi-gpu', action='store_true',
                   help='Distribute TTA views across all visible GPUs. '
                        'Each GPU gets its own model replica and processes '
                        'a subset of scale/flip views in parallel.')

    # Label-free prediction mode
    p.add_argument('--predict', action='store_true',
                   help='Prediction-only mode: skip metrics, output CSV. '
                        'Works with or without labels in parquet.')
    p.add_argument('--image-dir', default=None,
                   help='Image folder for prediction (no parquet needed). '
                        'All .jpg/.png files in the folder will be processed.')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Classification threshold for --predict mode '
                        '(default: 0.5)')
    return p.parse_args()


def _run_inference(args, df_or_dir, image_root, device):
    """Dispatch to the correct inference pipeline and return results."""
    cfg0 = Config.fromfile(args.config[0])
    if args.tta_scales:
        explicit_scales = args.tta_scales
    else:
        explicit_scales = cfg0.get('tta_scales', None)

    multigpu = args.multi_gpu and torch.cuda.device_count() > 1

    if args.ensemble and len(args.checkpoint) > 1:
        if len(args.config) == len(args.checkpoint):
            return multi_arch_ensemble(
                args.config, args.checkpoint, df_or_dir, image_root, device,
                use_tta=args.tta,
                batch_size=args.batch_size, num_workers=args.num_workers,
                tta_extra_scales=args.tta_extra_scales,
                tta_explicit_scales=explicit_scales,
                multigpu=multigpu)
        elif len(args.config) == 1:
            return ensemble_inference(
                args.config[0], args.checkpoint, df_or_dir, image_root,
                device, use_tta=args.tta,
                batch_size=args.batch_size, num_workers=args.num_workers,
                tta_extra_scales=args.tta_extra_scales,
                tta_explicit_scales=explicit_scales,
                multigpu=multigpu)
        else:
            raise ValueError(
                f'Number of configs ({len(args.config)}) must be 1 '
                f'or match number of checkpoints ({len(args.checkpoint)})')

    ckpt = args.checkpoint[0]

    if args.tta and multigpu:
        img_size = cfg0.get('img_size', 518)
        patch_size = cfg0.get('patch_size', 14)
        return tta_inference_multigpu(
            args.config[0], ckpt, df_or_dir, image_root,
            img_size, patch_size,
            batch_size=args.batch_size, num_workers=args.num_workers,
            extra_scales=args.tta_extra_scales,
            explicit_scales=explicit_scales)

    model, _, img_size, patch_size = build_model(
        args.config[0], ckpt, device)
    if args.tta:
        return tta_inference(
            model, df_or_dir, image_root, img_size, patch_size, device,
            batch_size=args.batch_size, num_workers=args.num_workers,
            extra_scales=args.tta_extra_scales,
            explicit_scales=explicit_scales)
    return standard_inference(
        model, df_or_dir, image_root, img_size, device,
        batch_size=args.batch_size, num_workers=args.num_workers)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(f'Device: {device}  (GPUs visible: {num_gpus})', flush=True)
    if args.multi_gpu and num_gpus > 1:
        print(f'Multi-GPU enabled: TTA views will be distributed across '
              f'{num_gpus} GPUs', flush=True)
    elif args.multi_gpu and num_gpus <= 1:
        print('Warning: --multi-gpu requested but only 1 GPU visible, '
              'falling back to single-GPU', flush=True)

    cfg = Config.fromfile(args.config[0])
    data_root    = args.data_root    or cfg.get('data_root',    'release')
    parquet_root = args.parquet_root or cfg.get('parquet_root', 'datasets')

    # ---- Determine data source ----
    if args.image_dir:
        # Pure image-folder prediction (no parquet)
        image_root = args.image_dir
        df_or_dir = args.image_dir  # signals ImageFolderDataset
        df = None
        print(f'\n=== Prediction from image folder ===')
        print(f'Images  : {image_root}', flush=True)
    else:
        parquet_path = os.path.join(
            parquet_root, f'query-{args.split}.parquet')
        image_root = os.path.join(
            data_root, 'data', args.split, 'query-data', 'crop')
        df = pd.read_parquet(parquet_path)
        df_or_dir = df
        has_labels = 'defect' in df.columns
        mode_str = 'Evaluation' if (has_labels and not args.predict) \
            else 'Prediction'
        print(f'\n=== {mode_str} on [{args.split}] split ===')
        print(f'Parquet : {parquet_path}')
        print(f'Images  : {image_root}')
        print(f'Samples : {len(df)}', flush=True)

    # ---- Run inference ----
    labels, probs, capture_ids = _run_inference(
        args, df_or_dir, image_root, device)

    has_valid_labels = (labels is not None
                        and len(labels) > 0
                        and (labels >= 0).all())

    # ---- Prediction-only mode ----
    if args.predict or not has_valid_labels:
        mode = 'predict'
        if args.tta:
            mode += '_tta'
        if args.ensemble:
            mode += '_ensemble'

        csv_path = os.path.join(
            args.output_dir, f'pred_{args.split}_{mode}.csv')
        out_df = pd.DataFrame(dict(
            capture_id=capture_ids,
            prob_defective=probs,
            pred=(probs >= args.threshold).astype(int),
        ))
        out_df.to_csv(csv_path, index=False)

        print(f'\n{"=" * 64}')
        print(f'  Prediction complete — {len(probs)} samples')
        print(f'  Threshold: {args.threshold}')
        print(f'  Predicted defective: '
              f'{(probs >= args.threshold).sum()} / {len(probs)}')
        print(f'{"=" * 64}')
        print(f'Predictions → {csv_path}')
        print('\nDone.', flush=True)
        return

    # ---- Full evaluation mode (labels available) ----
    metrics = compute_metrics(labels, probs)
    print('\n' + '=' * 64)
    print(f'  Results on [{args.split}]')
    print('=' * 64)
    print(f"  AP_any:           {metrics['AP_any (%)']:.2f}%")
    print(f"  AUROC:            {metrics['AUROC (%)']:.2f}%")
    print(f"  Recall@50%Prec:   {metrics['Recall@50%Precision (%)']:.2f}%")
    print(f"  Recall@1%FPR:     {metrics['Recall@1%FPR (%)']:.2f}%")
    print(f"  Optimal thresh:   {metrics['optimal_threshold']}")
    print('-' * 64)
    print(metrics['classification_report'])
    print(f"Confusion matrix:\n{np.array(metrics['confusion_matrix'])}")

    if df is not None:
        major = evaluate_major_only(df, probs)
        if major:
            print(f"\n  AP_major:         {major['AP_major']:.2f}%")
            metrics['AP_major'] = major

        mat_results = evaluate_by_material(df, probs, labels)
        if mat_results:
            print('\nPer-material AP:')
            for mat, info in mat_results.items():
                print(f"  {mat:25s}  AP={info['AP']:6.2f}%  "
                      f"(n={info['count']}, defective={info['defective']})")
            ap_values = [info['AP'] for info in mat_results.values()]
            mat_ap_std = float(np.std(ap_values))
            mat_ap_min = float(np.min(ap_values))
            mat_ap_mean = float(np.mean(ap_values))
            worst_mat = min(mat_results, key=lambda m: mat_results[m]['AP'])
            print(f"\n  Material AP std:  {mat_ap_std:.2f}  "
                  f"(mean={mat_ap_mean:.2f}%, min={mat_ap_min:.2f}% "
                  f"[{worst_mat}])")
            metrics['per_material'] = mat_results
            metrics['material_AP_std'] = round(mat_ap_std, 2)
            metrics['material_AP_min'] = round(mat_ap_min, 2)
            metrics['material_AP_mean'] = round(mat_ap_mean, 2)
            metrics['worst_material'] = worst_mat

    # ---- Save ----
    mode = 'tta' if args.tta else 'std'
    if args.ensemble:
        mode += '_ensemble'
    if len(args.config) > 1:
        mode += '_multiarch'

    json_path = os.path.join(args.output_dir, f'eval_{args.split}_{mode}.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f'\nMetrics → {json_path}')

    csv_path = os.path.join(args.output_dir, f'pred_{args.split}_{mode}.csv')
    pd.DataFrame(dict(
        capture_id=capture_ids,
        label=labels,
        prob_defective=probs,
        pred=(probs >= metrics['optimal_threshold']).astype(int),
    )).to_csv(csv_path, index=False)
    print(f'Predictions → {csv_path}')

    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
