"""
Kaputt Defect Detection — MMPretrain Evaluation (More TTA)
==========================================================
Extended TTA with 90°/180°/270° rotations.

Each scale produces 6 views:
  original + h_flip + v_flip + rot90 + rot180 + rot270

Total views = 6 × N_scales

Based on evaluate_vit_mm_mgpu.py with enhanced augmentation.

Usage:
    # TTA evaluation (multi-scale + flip + rotation)
    python evaluate_vit_mm_moretta.py --config configs/kaputt_dinov2.py \
        --checkpoint work_dirs/best.pth --tta

    # Multi-GPU TTA
    python evaluate_vit_mm_moretta.py --config configs/kaputt_dinov2.py \
        --checkpoint best.pth --tta --multi-gpu

    # Ensemble + TTA on multiple GPUs
    python evaluate_vit_mm_moretta.py \
        --config configs/kaputt_dinov2.py configs/kaputt_convnext.py \
        --checkpoint dinov2_best.pth convnext_best.pth \
        --ensemble --tta --multi-gpu \
        --tta-scales 512 752 1008 1248

    # Prediction mode
    python evaluate_vit_mm_moretta.py --config configs/kaputt_dinov2.py \
        --checkpoint best.pth --predict --image-dir /path/to/images/ --tta
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


def get_rotation_transform(img_size, angle):
    """Create a transform that resizes then rotates by exactly angle degrees.

    Args:
        img_size: Target spatial size.
        angle: Rotation angle in degrees (90, 180, 270).
    """
    ops = [
        transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(lambda img: img.rotate(-angle, expand=False)),
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
    """Single-pass inference returning P(defective) array.

    Uses CUDA streams to overlap data transfer and compute.
    """
    probs_list = []
    transfer_stream = torch.cuda.Stream(device=device)

    batches = iter(loader)
    # Prefetch first batch
    try:
        next_images, next_labels = next(batches)
    except StopIteration:
        return np.array([], dtype=np.float32)

    with torch.cuda.stream(transfer_stream):
        next_images = next_images.to(device, non_blocking=True)

    for images_cpu, _ in tqdm(batches, desc='  infer', leave=False,
                              file=sys.stdout, disable=quiet,
                              total=len(loader) - 1):
        # Wait for previous transfer to complete
        torch.cuda.current_stream(device).wait_stream(transfer_stream)
        images = next_images

        # Start transferring next batch while computing
        with torch.cuda.stream(transfer_stream):
            next_images = images_cpu.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
            logits = model(images, mode='tensor')
        probs = F.softmax(logits.float(), dim=1)[:, 1]
        probs_list.append(probs.cpu())

    # Process last prefetched batch
    torch.cuda.current_stream(device).wait_stream(transfer_stream)
    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
        logits = model(next_images, mode='tensor')
    probs = F.softmax(logits.float(), dim=1)[:, 1]
    probs_list.append(probs.cpu())

    return torch.cat(probs_list).numpy()


_PREFETCH_FACTOR = 8  # overridden by CLI --prefetch-factor


def _make_loader(ds, batch_size, num_workers, prefetch_factor=None,
                 persistent_workers=True):
    """Create a DataLoader with aggressive prefetching to keep GPU fed."""
    pf = prefetch_factor if prefetch_factor is not None else _PREFETCH_FACTOR
    pw = persistent_workers and num_workers > 0
    pf_arg = pf if num_workers > 0 else None
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=pf_arg, persistent_workers=pw,
        drop_last=False,
    )


def _make_dataset(df_or_dir, image_root, transform):
    """Create dataset from either a DataFrame+image_root or image directory."""
    if isinstance(df_or_dir, pd.DataFrame):
        return SimpleKaputtDataset(df_or_dir, image_root, transform)
    return ImageFolderDataset(df_or_dir, transform)


def standard_inference(model, df_or_dir, image_root, img_size, device,
                       batch_size=16, num_workers=8):
    """Standard single-scale inference."""
    ds = _make_dataset(df_or_dir, image_root, get_eval_transform(img_size))
    loader = _make_loader(ds, batch_size, num_workers)
    probs = _inference_pass(model, loader, device)
    labels = np.array(ds.labels)
    capture_ids = ds.capture_ids
    return labels, probs, capture_ids


def tta_inference(model, df_or_dir, image_root, base_size, patch_size, device,
                  batch_size=16, num_workers=8,
                  extra_scales=(0, 2, 4), use_flip=True,
                  explicit_scales=None):
    """Multi-scale + flip + rotation TTA.

    For each scale, 6 views are generated:
      1. original
      2. horizontal flip
      3. vertical flip
      4. 90° rotation
      5. 180° rotation
      6. 270° rotation

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

    # Build all (scale, aug_type) view descriptors upfront
    view_specs = []
    for s in scales:
        view_specs.append((s, None))           # original
        if use_flip:
            view_specs.append((s, 'horizontal'))
            view_specs.append((s, 'vertical'))
        view_specs.append((s, 'rot90'))
        view_specs.append((s, 'rot180'))
        view_specs.append((s, 'rot270'))

    def _get_transform(scale, aug_type):
        if aug_type is None:
            return get_eval_transform(scale)
        elif aug_type in ('horizontal', 'vertical'):
            return get_flip_transform(scale, aug_type)
        elif aug_type == 'rot90':
            return get_rotation_transform(scale, 90)
        elif aug_type == 'rot180':
            return get_rotation_transform(scale, 180)
        elif aug_type == 'rot270':
            return get_rotation_transform(scale, 270)
        return get_eval_transform(scale)

    # Prefetch pipeline: prepare next view's DataLoader in background thread
    # while current view is running inference on GPU
    from concurrent.futures import ThreadPoolExecutor

    def _prepare_loader(spec):
        s, aug_type = spec
        tfm = _get_transform(s, aug_type)
        ds = _make_dataset(df_or_dir, image_root, tfm)
        loader = _make_loader(ds, batch_size, num_workers)
        return ds, loader

    prefetch_pool = ThreadPoolExecutor(max_workers=2)

    # Submit first two loaders
    futures = []
    for spec in view_specs[:2]:
        futures.append(prefetch_pool.submit(_prepare_loader, spec))
    next_submit_idx = 2

    for vi, spec in enumerate(view_specs):
        s, aug_type = spec
        tag = f'{s}×{s}'
        if aug_type:
            tag += f'+{aug_type}'
        print(f'  TTA [{vi+1}/{len(view_specs)}] {tag}', flush=True)

        # Get current loader from prefetch
        ds, loader = futures[vi].result()

        # Submit next view's loader preparation while GPU works
        if next_submit_idx < len(view_specs):
            futures.append(prefetch_pool.submit(
                _prepare_loader, view_specs[next_submit_idx]))
            next_submit_idx += 1

        all_probs.append(_inference_pass(model, loader, device))
        if labels is None:
            labels = np.array(ds.labels)
            capture_ids = ds.capture_ids

    prefetch_pool.shutdown(wait=False)

    print(f'  TTA views: {len(all_probs)} '
          f'(6 augmentations × {len(scales)} scales)', flush=True)
    avg = np.mean(all_probs, axis=0)
    return labels, avg, capture_ids


def tta_inference_multigpu(config_path, checkpoint_path, df_or_dir, image_root,
                           base_size, patch_size, batch_size=16, num_workers=8,
                           extra_scales=(0, 2, 4), use_flip=True,
                           explicit_scales=None):
    """Multi-GPU TTA: distribute scale/flip/rotation views across GPUs.

    Each GPU gets its own model replica and processes a subset of TTA views
    (scale × {original, h_flip, v_flip, rot90, rot180, rot270}) in parallel.
    Falls back to single-GPU ``tta_inference`` when only one GPU is visible.
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

    # Build all views: (scale, augmentation_type)
    # augmentation_type: None=original, 'horizontal', 'vertical',
    #                    'rot90', 'rot180', 'rot270'
    views = []
    for s in scales:
        views.append((s, None))          # original
        if use_flip:
            views.append((s, 'horizontal'))  # h_flip
            views.append((s, 'vertical'))    # v_flip
        views.append((s, 'rot90'))       # 90° rotation
        views.append((s, 'rot180'))      # 180° rotation
        views.append((s, 'rot270'))      # 270° rotation

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

    workers_per_gpu = max(2, num_workers // num_gpus)
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

    def _get_transform(scale, aug_type):
        if aug_type is None:
            return get_eval_transform(scale)
        elif aug_type in ('horizontal', 'vertical'):
            return get_flip_transform(scale, aug_type)
        elif aug_type == 'rot90':
            return get_rotation_transform(scale, 90)
        elif aug_type == 'rot180':
            return get_rotation_transform(scale, 180)
        elif aug_type == 'rot270':
            return get_rotation_transform(scale, 270)
        return get_eval_transform(scale)

    def _gpu_worker(gpu_id):
        model, device = models[gpu_id]
        task_list = gpu_tasks[gpu_id]

        # Prefetch: prepare next view's loader while current one runs
        def _prep(view_item):
            _, (scale, aug_type) = view_item
            tfm = _get_transform(scale, aug_type)
            ds = _make_dataset(df_or_dir, image_root, tfm)
            return _make_loader(ds, batch_size, workers_per_gpu,
                                prefetch_factor=4, persistent_workers=False)

        from concurrent.futures import ThreadPoolExecutor as _TPE
        prefetch = _TPE(max_workers=1)
        next_fut = prefetch.submit(_prep, task_list[0]) if task_list else None

        for ti, (view_idx, (scale, aug_type)) in enumerate(task_list):
            tag = f'{scale}×{scale}'
            if aug_type:
                tag += f'+{aug_type}'
            print(f'  [GPU {gpu_id}] TTA {tag} ...', flush=True)
            t0 = time.perf_counter()

            loader = next_fut.result()

            # Start prefetching next view's loader
            if ti + 1 < len(task_list):
                next_fut = prefetch.submit(_prep, task_list[ti + 1])

            all_probs[view_idx] = _inference_pass(
                model, loader, device, quiet=True)
            elapsed = time.perf_counter() - t0
            with counter_lock:
                done_counter[0] += 1
                print(f'  [GPU {gpu_id}] TTA {tag} done  '
                      f'({elapsed:.1f}s)  '
                      f'[{done_counter[0]}/{total_views} views]',
                      flush=True)
        prefetch.shutdown(wait=False)

    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futs = [pool.submit(_gpu_worker, gid) for gid in range(num_gpus)]
        for f in futs:
            f.result()

    del models
    torch.cuda.empty_cache()

    print(f'  TTA views: {len(views)} '
          f'(6 augmentations × {len(scales)} scales)', flush=True)
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
        description='Kaputt MMPretrain — Evaluation / More TTA (6-aug) / Ensemble')
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
    p.add_argument('--num-workers', type=int, default=16)
    p.add_argument('--prefetch-factor', type=int, default=6,
                   help='Number of batches each DataLoader worker prefetches '
                        '(higher = more CPU memory, less GPU starvation)')
    p.add_argument('--output-dir', default='results_vit_mm')

    p.add_argument('--tta', action='store_true',
                   help='Enable multi-scale + flip + rotation TTA (6 views/scale)')
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
                        'a subset of scale/flip/rotation views in parallel.')

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
    global _PREFETCH_FACTOR
    args = parse_args()
    _PREFETCH_FACTOR = args.prefetch_factor
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(f'Device: {device}  (GPUs visible: {num_gpus})', flush=True)
    print(f'DataLoader: num_workers={args.num_workers}, '
          f'prefetch_factor={_PREFETCH_FACTOR}', flush=True)
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
        image_root = args.image_dir
        df_or_dir = args.image_dir
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

    if args.tta:
        print('TTA mode: 6 augmentations per scale '
              '(original + h_flip + v_flip + rot90 + rot180 + rot270)',
              flush=True)

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
            mode += '_tta6'
        if args.ensemble:
            mode += '_ensemble'

        csv_path = os.path.join(
            args.output_dir, f'pred_{args.split}_{mode}.csv')
        out_df = pd.DataFrame(dict(
            capture_id=capture_ids,
            pred=probs,
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
    mode = 'tta6' if args.tta else 'std'
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
        pred=probs,
    )).to_csv(csv_path, index=False)
    print(f'Predictions → {csv_path}')

    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
