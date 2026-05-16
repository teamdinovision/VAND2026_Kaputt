"""
Reference-Enhanced Inference for Kaputt Defect Detection
=========================================================
Leverages reference images at INFERENCE TIME (no training) to
complement the trained classifier's predictions.

Strategy rationale:
  The paper shows naive reference usage during TRAINING hurts
  performance (90.67% → 71.21% AP). But at INFERENCE time, with
  a strong pretrained classifier already in place, reference images
  provide two complementary signals:

  A. Reference-Calibrated Probability
     Run the classifier on reference images too. If the classifier
     thinks the references are also "defective", it's likely the item
     just looks unusual — reduce the query's defect score (fewer FP).

  B. CLS-Token Cosine Anomaly
     DINOv3-L features are highly semantic and partially pose-invariant.
     Large cosine distance between query and reference CLS tokens
     suggests the query differs from normal — potential defect.

  C. Patch-Level Anomaly (optional, at reduced resolution)
     PatchCore-style: for each query patch, find the closest reference
     patch. Large distances indicate local anomalies.

  All signals are rank-normalized and fused with configurable weights.
  The user can validate on Kaputt1 test (with labels) to find optimal
  alpha, then apply to Kaputt2.

Usage:
    # Validate on Kaputt1 test set (HAS labels — see which alpha helps)
    python evaluate_with_ref.py \\
        --config configs/large_728_three_freeze.py \\
        --checkpoint work_dirs/best.pth \\
        --kaputt-root /data/public/dataset/kaputt \\
        --output-dir results_ref_k1

    # Run on Kaputt2 competition data (NO labels)
    python evaluate_with_ref.py \\
        --config configs/large_728_three_freeze.py \\
        --checkpoint work_dirs/best.pth \\
        --kaputt-root /data/public/dataset/kaputt2 \\
        --output-dir results_ref_k2

    # With TTA for classifier probabilities
    python evaluate_with_ref.py \\
        --config configs/large_728_three_freeze.py \\
        --checkpoint work_dirs/best.pth \\
        --kaputt-root /data/public/dataset/kaputt2 \\
        --tta --tta-scales 512 728 960 \\
        --output-dir results_ref_k2_tta

    # Multi-GPU TTA
    python evaluate_with_ref.py \\
        --config configs/large_728_three_freeze.py \\
        --checkpoint work_dirs/best.pth \\
        --kaputt-root /data/public/dataset/kaputt2 \\
        --tta --tta-scales 512 728 960 --multi-gpu \\
        --output-dir results_ref_k2_tta
"""

import os
import sys
import argparse
import time
import threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmpretrain_kaputt  # noqa: F401

from mmengine.config import Config
from mmpretrain.registry import MODELS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ------------------------------------------------------------------
# Datasets
# ------------------------------------------------------------------
class PathListDataset(Dataset):
    """Load images from a list of absolute paths."""

    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, idx


class CaptureIDDataset(Dataset):
    """Load images from a directory using capture_id as filename."""

    def __init__(self, capture_ids, image_dir, transform=None):
        self.capture_ids = capture_ids
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.capture_ids)

    def __getitem__(self, idx):
        img = Image.open(
            self.image_dir / f'{self.capture_ids[idx]}.jpg').convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, idx


class PreloadedDataset(Dataset):
    """Images pre-loaded in memory — eliminates repeated disk I/O for TTA.

    Stores decoded images as a list of PIL Images.  Since DataLoader
    workers with fork start-method share the parent's memory via
    copy-on-write, the image list is NOT duplicated per worker as long
    as the list itself is not mutated in workers.  The transform
    (resize/flip/normalize) creates new tensors without touching the
    stored PIL objects, so COW is preserved.

    For ``num_workers=0`` (main-process loading) this is always safe.
    For ``num_workers>0`` on Linux (fork), COW sharing keeps memory flat.
    """

    def __init__(self, pil_images, transform=None):
        self.images = pil_images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, idx


def preload_images_by_capture_id(capture_ids, image_dir, num_workers=8):
    """Read all images from disk into a list of PIL Images (one-time cost).

    Uses ThreadPoolExecutor to parallelise JPEG decoding across CPU cores.
    """
    image_dir = Path(image_dir)

    def _load_one(cid):
        return Image.open(image_dir / f'{cid}.jpg').convert('RGB')

    print(f'  Preloading {len(capture_ids)} images into RAM ...', flush=True)
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        images = list(tqdm(
            pool.map(_load_one, capture_ids),
            total=len(capture_ids), desc='  preload', leave=False))

    elapsed = time.perf_counter() - t0
    mem_mb = sum(img.size[0] * img.size[1] * 3 for img in images) / 1e6
    print(f'  Preloaded {len(images)} images in {elapsed:.1f}s '
          f'(~{mem_mb:.0f} MB in RAM)', flush=True)
    return images


def preload_images_by_path(paths, num_workers=8):
    """Read all images from disk into a list of PIL Images (one-time cost)."""

    def _load_one(p):
        return Image.open(p).convert('RGB')

    print(f'  Preloading {len(paths)} ref images into RAM ...', flush=True)
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        images = list(tqdm(
            pool.map(_load_one, paths),
            total=len(paths), desc='  preload', leave=False))

    elapsed = time.perf_counter() - t0
    mem_mb = sum(img.size[0] * img.size[1] * 3 for img in images) / 1e6
    print(f'  Preloaded {len(images)} images in {elapsed:.1f}s '
          f'(~{mem_mb:.0f} MB in RAM)', flush=True)
    return images


# ------------------------------------------------------------------
# Model building
# ------------------------------------------------------------------
def build_model(config_path, checkpoint_path, device):
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


def get_transform(img_size):
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
    else:
        ops.append(transforms.RandomVerticalFlip(p=1.0))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def get_rotation_transform(img_size, angle):
    """Build a transform that resizes then rotates by a fixed angle (90/180/270)."""
    return transforms.Compose([
        transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(
            lambda img, a=angle: transforms.functional.rotate(img, a)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_diagonal_flip_transform(img_size, direction='transpose'):
    """Build a transform that resizes then flips along a diagonal.

    These two transforms, together with the 4 rotations + 2 axis flips,
    complete the D4 dihedral group (all 8 rigid symmetries of a square).

    Args:
        direction: 'transpose'  — reflect across the main diagonal (top-left ↔ bottom-right)
                   'transverse' — reflect across the anti-diagonal (top-right ↔ bottom-left)
    """
    if direction == 'transpose':
        flip_method = Image.TRANSPOSE
    else:
        flip_method = Image.TRANSVERSE
    return transforms.Compose([
        transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(lambda img, m=flip_method: img.transpose(m)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_shift_transform(img_size, dx, dy):
    """Resize to img_size + pad, then crop with (dx, dy) pixel offset.

    Shifts the patch grid alignment for ViT.  The image is first resized
    to (img_size + 2*|max_shift|) so that after shifting we can crop
    back to exactly img_size without losing content or adding black borders.
    """
    pad = max(abs(dx), abs(dy))
    padded = img_size + 2 * pad

    def _shift_crop(img, _dx=dx, _dy=dy, _pad=pad, _sz=img_size):
        left = _pad + _dx
        top = _pad + _dy
        return img.crop((left, top, left + _sz, top + _sz))

    return transforms.Compose([
        transforms.Resize(
            (padded, padded),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(_shift_crop),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_center_crop_transform(img_size, crop_ratio):
    """Resize larger, then center-crop to img_size — effective zoom-in."""
    upscale = int(round(img_size / crop_ratio))
    return transforms.Compose([
        transforms.Resize(
            (upscale, upscale),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_five_crop_transform(img_size, position):
    """Resize larger then take one of 5 crops (4 corners + center).

    Resizes to ~1.15x img_size so each crop covers ~75% of area,
    providing meaningful spatial diversity.
    """
    upscale = int(round(img_size * 1.15))

    def _crop(img, _pos=position, _sz=img_size):
        w, h = img.size
        if _pos == 'center':
            left = (w - _sz) // 2
            top = (h - _sz) // 2
        elif _pos == 'tl':
            left, top = 0, 0
        elif _pos == 'tr':
            left, top = w - _sz, 0
        elif _pos == 'bl':
            left, top = 0, h - _sz
        elif _pos == 'br':
            left, top = w - _sz, h - _sz
        else:
            left = (w - _sz) // 2
            top = (h - _sz) // 2
        return img.crop((left, top, left + _sz, top + _sz))

    return transforms.Compose([
        transforms.Resize(
            (upscale, upscale),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(_crop),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ------------------------------------------------------------------
# DataLoader factory with aggressive prefetching
# ------------------------------------------------------------------
_PREFETCH_FACTOR = 8


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


# ------------------------------------------------------------------
# Feature + logit extraction (single forward pass via hook)
# ------------------------------------------------------------------
@torch.no_grad()
def extract_features_and_probs(model, loader, device,
                               amp_dtype=torch.bfloat16):
    """Extract CLS features and classification probabilities in one pass.

    Uses a forward hook on model.backbone to capture raw ViT output
    (CLS + patch tokens) while still running the full classifier forward.
    Uses CUDA streams to overlap data transfer with compute.

    Returns:
        cls_features: (N, D) tensor — CLS token from backbone
        probs:        (N,) ndarray  — P(defective)
    """
    captured = []

    def hook_fn(module, inp, out):
        feat = out[-1] if isinstance(out, (tuple, list)) else out
        captured.append(feat.detach())

    handle = model.backbone.register_forward_hook(hook_fn)

    cls_list, prob_list = [], []
    transfer_stream = torch.cuda.Stream(device=device)

    batches = iter(loader)
    try:
        next_images, _ = next(batches)
    except StopIteration:
        handle.remove()
        return torch.zeros(0), np.array([], dtype=np.float32)

    with torch.cuda.stream(transfer_stream):
        next_images = next_images.to(device, non_blocking=True)

    for images_cpu, _ in tqdm(batches, desc='  extract', leave=False,
                              total=len(loader) - 1):
        torch.cuda.current_stream(device).wait_stream(transfer_stream)
        images = next_images

        with torch.cuda.stream(transfer_stream):
            next_images = images_cpu.to(device, non_blocking=True)

        captured.clear()
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
            logits = model(images, mode='tensor')

        feat = captured[0]
        cls_tok = feat[:, 0, :] if feat.dim() == 3 else feat
        cls_list.append(cls_tok.float().cpu())
        prob_list.append(F.softmax(logits.float(), dim=1)[:, 1].cpu())

    # Process last prefetched batch
    torch.cuda.current_stream(device).wait_stream(transfer_stream)
    captured.clear()
    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
        logits = model(next_images, mode='tensor')
    feat = captured[0]
    cls_tok = feat[:, 0, :] if feat.dim() == 3 else feat
    cls_list.append(cls_tok.float().cpu())
    prob_list.append(F.softmax(logits.float(), dim=1)[:, 1].cpu())

    handle.remove()

    cls_features = torch.cat(cls_list, dim=0)
    probs = torch.cat(prob_list, dim=0).numpy()
    return cls_features, probs


@torch.no_grad()
def inference_probs_only(model, loader, device, amp_dtype=torch.bfloat16,
                         quiet=False):
    """Forward pass with CUDA stream prefetch, returning P(defective)."""
    probs_list = []
    transfer_stream = torch.cuda.Stream(device=device)

    batches = iter(loader)
    try:
        next_images, _ = next(batches)
    except StopIteration:
        return np.array([], dtype=np.float32)

    with torch.cuda.stream(transfer_stream):
        next_images = next_images.to(device, non_blocking=True)

    for images_cpu, _ in tqdm(batches, desc='  infer', leave=False,
                              total=len(loader) - 1, disable=quiet):
        torch.cuda.current_stream(device).wait_stream(transfer_stream)
        images = next_images

        with torch.cuda.stream(transfer_stream):
            next_images = images_cpu.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
            logits = model(images, mode='tensor')
        probs_list.append(F.softmax(logits.float(), dim=1)[:, 1].cpu())

    # Process last prefetched batch
    torch.cuda.current_stream(device).wait_stream(transfer_stream)
    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
        logits = model(next_images, mode='tensor')
    probs_list.append(F.softmax(logits.float(), dim=1)[:, 1].cpu())

    return torch.cat(probs_list).numpy()


# ------------------------------------------------------------------
# TTA (self-contained, mirrors evaluate_vit_mm.py logic)
# ------------------------------------------------------------------
def _compute_scale_weights(scales, views_per_scale, center=None, sigma=0.3):
    """Gaussian weights based on proximity to center scale."""
    if center is None:
        center = float(np.mean(scales))
    weights = []
    for s in scales:
        ratio = (s - center) / center
        w = np.exp(-ratio**2 / (2 * sigma**2))
        for _ in range(views_per_scale):
            weights.append(w)
    weights = np.array(weights)
    weights /= weights.sum()
    return weights


def _build_view_specs(scales, use_flip=True, use_rotation=True,
                      use_diagonal=True, use_shift=False,
                      shift_pixels=None, use_crop=False,
                      crop_ratios=None, use_five_crop=False,
                      patch_size=14):
    """Build list of (scale, aug_type) tuples for TTA.

    aug_type is one of:
      D4 group:  None, 'horizontal', 'vertical',
                 'rot90', 'rot180', 'rot270', 'transpose', 'transverse'
      Shift:     'shift_dx_dy' (e.g. 'shift_7_0', 'shift_0_7', 'shift_7_7')
      Crop:      'crop_0.875', 'crop_0.9', ...
      Five-crop: 'fivecrop_tl', 'fivecrop_tr', 'fivecrop_bl', 'fivecrop_br',
                 'fivecrop_center'
    """
    if shift_pixels is None:
        shift_pixels = [patch_size // 2]

    if crop_ratios is None:
        crop_ratios = [0.875, 0.95]

    view_specs = []
    for s in scales:
        # --- D4 rigid symmetries ---
        view_specs.append((s, None))
        if use_flip:
            view_specs.append((s, 'horizontal'))
            view_specs.append((s, 'vertical'))
        if use_rotation:
            view_specs.append((s, 'rot90'))
            view_specs.append((s, 'rot180'))
            view_specs.append((s, 'rot270'))
        if use_diagonal:
            view_specs.append((s, 'transpose'))
            view_specs.append((s, 'transverse'))
        # --- Sub-patch shift ---
        if use_shift:
            for d in shift_pixels:
                view_specs.append((s, f'shift_{d}_0'))
                view_specs.append((s, f'shift_0_{d}'))
                view_specs.append((s, f'shift_{d}_{d}'))
                view_specs.append((s, f'shift_{-d}_{-d}'))
        # --- Center crop (zoom-in) ---
        if use_crop:
            for r in crop_ratios:
                view_specs.append((s, f'crop_{r}'))
        # --- Five-crop ---
        if use_five_crop:
            for pos in ('tl', 'tr', 'bl', 'br', 'center'):
                view_specs.append((s, f'fivecrop_{pos}'))
    return view_specs


def _get_tta_transform(scale, aug_type):
    """Return the transform for a given TTA view."""
    if aug_type is None:
        return get_transform(scale)
    if aug_type in ('horizontal', 'vertical'):
        return get_flip_transform(scale, aug_type)
    if aug_type in ('transpose', 'transverse'):
        return get_diagonal_flip_transform(scale, aug_type)
    if aug_type.startswith('shift_'):
        parts = aug_type.split('_')
        dx, dy = int(parts[1]), int(parts[2])
        return get_shift_transform(scale, dx, dy)
    if aug_type.startswith('crop_'):
        ratio = float(aug_type.split('_')[1])
        return get_center_crop_transform(scale, ratio)
    if aug_type.startswith('fivecrop_'):
        pos = aug_type.split('_', 1)[1]
        return get_five_crop_transform(scale, pos)
    angle = {'rot90': 90, 'rot180': 180, 'rot270': 270}[aug_type]
    return get_rotation_transform(scale, angle)


def _count_views_per_scale(use_flip, use_rotation, use_diagonal=True,
                           use_shift=False, shift_pixels=None,
                           use_crop=False, crop_ratios=None,
                           use_five_crop=False, patch_size=14):
    """Number of TTA views generated per scale."""
    if shift_pixels is None:
        shift_pixels = [patch_size // 2]
    if crop_ratios is None:
        crop_ratios = [0.875, 0.95]

    n = 1
    if use_flip:
        n += 2
    if use_rotation:
        n += 3
    if use_diagonal:
        n += 2
    if use_shift:
        n += 4 * len(shift_pixels)
    if use_crop:
        n += len(crop_ratios)
    if use_five_crop:
        n += 5
    return n


def tta_inference(model, dataset_factory, img_size, patch_size, device,
                  batch_size=64, num_workers=8, explicit_scales=None,
                  use_flip=True, use_rotation=True, use_diagonal=True,
                  use_shift=False, shift_pixels=None,
                  use_crop=False, crop_ratios=None,
                  use_five_crop=False,
                  weighted=False, tta_center=None,
                  tta_sigma=0.3):
    """Multi-scale TTA with D4, shift, crop, and five-crop augmentations.

    Returns averaged P(defective).

    Uses ThreadPoolExecutor to prepare the next view's DataLoader in a
    background thread while the current view runs on GPU, eliminating
    the gap between views.
    """
    if explicit_scales:
        scales = sorted(set(explicit_scales))
    else:
        base = (img_size // patch_size) * patch_size
        scales = sorted({base + d * patch_size for d in [0, 2, 4]})

    view_specs = _build_view_specs(
        scales, use_flip, use_rotation, use_diagonal,
        use_shift, shift_pixels, use_crop, crop_ratios,
        use_five_crop, patch_size)
    nv = _count_views_per_scale(
        use_flip, use_rotation, use_diagonal,
        use_shift, shift_pixels, use_crop, crop_ratios,
        use_five_crop, patch_size)

    def _prepare_loader(spec):
        s, aug_type = spec
        tfm = _get_tta_transform(s, aug_type)
        ds = dataset_factory(tfm)
        return _make_loader(ds, batch_size, num_workers,
                            persistent_workers=False)

    prefetch_pool = ThreadPoolExecutor(max_workers=2)

    futures = []
    for spec in view_specs[:2]:
        futures.append(prefetch_pool.submit(_prepare_loader, spec))
    next_submit_idx = 2

    all_probs = []
    for vi, spec in enumerate(view_specs):
        s, aug_type = spec
        tag = f'{s}x{s}'
        if aug_type:
            tag += f'+{aug_type}'
        print(f'  TTA [{vi+1}/{len(view_specs)}] {tag}', flush=True)

        loader = futures[vi].result()

        if next_submit_idx < len(view_specs):
            futures.append(prefetch_pool.submit(
                _prepare_loader, view_specs[next_submit_idx]))
            next_submit_idx += 1

        all_probs.append(inference_probs_only(model, loader, device))

    prefetch_pool.shutdown(wait=False)

    print(f'  TTA views: {len(all_probs)} '
          f'({len(scales)} scales x {nv} views/scale)', flush=True)

    if weighted:
        w = _compute_scale_weights(scales, nv, tta_center, tta_sigma)
        ctr = tta_center if tta_center else int(np.mean(scales))
        print(f'  Scale weights (center={ctr}, sigma={tta_sigma}):', flush=True)
        vi = 0
        for s in scales:
            print(f'    {s}px: weight={w[vi]:.4f} (x{nv} views)', flush=True)
            vi += nv
        return np.average(all_probs, axis=0, weights=w)
    return np.mean(all_probs, axis=0)


def tta_inference_multigpu(config_path, checkpoint_path,
                           dataset_factory, img_size, patch_size,
                           batch_size=256, num_workers=8,
                           explicit_scales=None, use_flip=True,
                           use_rotation=True, use_diagonal=True,
                           use_shift=False, shift_pixels=None,
                           use_crop=False, crop_ratios=None,
                           use_five_crop=False,
                           weighted=False, tta_center=None,
                           tta_sigma=0.3):
    """Multi-GPU TTA: distribute all view types across all GPUs.

    Each GPU gets its own model replica and processes a round-robin subset
    of TTA views in parallel.  Falls back to single-GPU ``tta_inference``
    when only one GPU is available.

    Returns:
        np.ndarray of averaged P(defective) per sample.
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        device = torch.device('cuda:0' if torch.cuda.is_available()
                              else 'cpu')
        model, _, _, _ = build_model(config_path, checkpoint_path, device)
        return tta_inference(
            model, dataset_factory, img_size, patch_size, device,
            batch_size=batch_size, num_workers=num_workers,
            explicit_scales=explicit_scales, use_flip=use_flip,
            use_rotation=use_rotation, use_diagonal=use_diagonal,
            use_shift=use_shift, shift_pixels=shift_pixels,
            use_crop=use_crop, crop_ratios=crop_ratios,
            use_five_crop=use_five_crop,
            weighted=weighted, tta_center=tta_center, tta_sigma=tta_sigma)

    # --- Compute scales & view specs ---
    if explicit_scales:
        scales = sorted(set(explicit_scales))
    else:
        base = (img_size // patch_size) * patch_size
        scales = sorted({base + d * patch_size for d in [0, 2, 4]})

    view_specs = _build_view_specs(
        scales, use_flip, use_rotation, use_diagonal,
        use_shift, shift_pixels, use_crop, crop_ratios,
        use_five_crop, patch_size)
    nv = _count_views_per_scale(
        use_flip, use_rotation, use_diagonal,
        use_shift, shift_pixels, use_crop, crop_ratios,
        use_five_crop, patch_size)

    workers_per_gpu = 0
    print(f'  Multi-GPU TTA: {len(view_specs)} views on {num_gpus} GPUs '
          f'({len(scales)} scales x {nv} views/scale, '
          f'in-thread dataloader, no fork)', flush=True)

    # --- Build one model replica per GPU ---
    models = []
    for gpu_id in range(num_gpus):
        dev = torch.device(f'cuda:{gpu_id}')
        m, _, _, _ = build_model(config_path, checkpoint_path, dev)
        models.append((m, dev))

    # --- Round-robin assignment of views to GPUs ---
    gpu_tasks = [[] for _ in range(num_gpus)]
    for i, spec in enumerate(view_specs):
        gpu_tasks[i % num_gpus].append((i, spec))

    all_probs = [None] * len(view_specs)
    total_views = len(view_specs)
    done_counter = [0]
    counter_lock = threading.Lock()

    def _gpu_worker(gpu_id):
        model, device = models[gpu_id]
        task_list = gpu_tasks[gpu_id]

        def _prep(view_item):
            _, (scale, aug_type) = view_item
            tfm = _get_tta_transform(scale, aug_type)
            ds = dataset_factory(tfm)
            return _make_loader(ds, batch_size, workers_per_gpu,
                                persistent_workers=False)

        prefetch = ThreadPoolExecutor(max_workers=1)
        next_fut = prefetch.submit(_prep, task_list[0]) if task_list else None

        for ti, (view_idx, (scale, aug_type)) in enumerate(task_list):
            tag = f'{scale}x{scale}'
            if aug_type:
                tag += f'+{aug_type}'
            t0 = time.perf_counter()

            loader = next_fut.result()
            if ti + 1 < len(task_list):
                next_fut = prefetch.submit(_prep, task_list[ti + 1])

            all_probs[view_idx] = inference_probs_only(
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

    print(f'  Multi-GPU TTA complete: {len(view_specs)} views', flush=True)

    if weighted:
        w = _compute_scale_weights(scales, nv, tta_center, tta_sigma)
        ctr = tta_center if tta_center else int(np.mean(scales))
        print(f'  Scale weights (center={ctr}, sigma={tta_sigma}):', flush=True)
        vi = 0
        for s in scales:
            print(f'    {s}px: weight={w[vi]:.4f} (x{nv} views)', flush=True)
            vi += nv
        return np.average(all_probs, axis=0, weights=w)
    return np.mean(all_probs, axis=0)


# ------------------------------------------------------------------
# Reference bank
# ------------------------------------------------------------------
def resolve_ref_paths(ref_df, ref_image_dir, data_root):
    """Parse reference parquet and resolve image paths.

    Returns list of (item_identifier, absolute_path) tuples.
    """
    ref_dir = Path(ref_image_dir)
    root = Path(data_root) if data_root else None
    pairs = []

    for _, row in ref_df.iterrows():
        item_id = row['item_identifier']
        crop_field = str(row.get('reference_crop', ''))
        if not crop_field or crop_field == 'nan':
            continue

        raw_paths = [p.strip() for p in crop_field.split(',')]
        for rp in raw_paths:
            if not rp:
                continue
            candidates = [
                ref_dir / Path(rp).name,
                ref_dir / rp,
            ]
            if root:
                candidates.append(root / rp)
            resolved = None
            for c in candidates:
                if c.exists():
                    resolved = str(c)
                    break
            if resolved:
                pairs.append((item_id, resolved))

    return pairs


def build_reference_bank(model, ref_pairs, img_size, device,
                         batch_size=512, num_workers=4,
                         multi_scale_sizes=None):
    """Build per-item CLS feature bank and classifier probs for references.

    Args:
        multi_scale_sizes: optional list of sizes for multi-scale feature
            extraction. Features from all scales are L2-averaged for more
            robust matching.

    Returns:
        item_cls:  dict[item_id] -> (num_refs, D) CLS features
        item_prob: dict[item_id] -> (num_refs,) P(defect) from classifier
    """
    if not ref_pairs:
        return {}, {}

    item_ids = [p[0] for p in ref_pairs]
    paths = [p[1] for p in ref_pairs]

    sizes = multi_scale_sizes if multi_scale_sizes else [img_size]

    # Preload ref images once, reuse across all scales
    pil_images = preload_images_by_path(paths, num_workers=num_workers)

    all_cls_feats = []
    all_probs = []
    for sz in sizes:
        print(f'  Extracting ref features at {sz}x{sz} '
              f'({len(paths)} images) ...', flush=True)
        tfm = get_transform(sz)
        ds = PreloadedDataset(pil_images, tfm)
        ld = _make_loader(ds, batch_size, num_workers)
        cls_feats, probs = extract_features_and_probs(model, ld, device)
        all_cls_feats.append(cls_feats)
        all_probs.append(probs)

    cls_feats = torch.stack(all_cls_feats).mean(dim=0)
    cls_feats = F.normalize(cls_feats, dim=1)
    probs = np.mean(all_probs, axis=0)

    item_cls = defaultdict(list)
    item_prob = defaultdict(list)
    for i, iid in enumerate(item_ids):
        item_cls[iid].append(cls_feats[i])
        item_prob[iid].append(probs[i])

    item_cls_t = {k: torch.stack(v) for k, v in item_cls.items()}
    item_prob_a = {k: np.array(v) for k, v in item_prob.items()}

    n_items = len(item_cls_t)
    avg_refs = np.mean([v.shape[0] for v in item_cls_t.values()])
    print(f'  Reference bank: {n_items} items, {avg_refs:.1f} refs/item '
          f'(scales={sizes})', flush=True)
    return item_cls_t, item_prob_a


# ------------------------------------------------------------------
# Anomaly scoring
# ------------------------------------------------------------------
def score_cls_anomaly(query_cls, ref_cls_bank, query_items, agg='max'):
    """CLS-token cosine distance anomaly score.

    Higher = more different from references = more likely defective.
    Items without references get the population median (neutral).

    Args:
        agg: aggregation over reference similarities.
             'max'  — use closest reference (original, conservative)
             'mean' — use mean distance to all references (stricter)
             'topk' — use mean of top-3 closest references
    """
    n = len(query_items)
    scores = np.full(n, np.nan)
    n_miss = 0

    for i, iid in enumerate(query_items):
        if iid not in ref_cls_bank:
            n_miss += 1
            continue
        ref = ref_cls_bank[iid]
        q = query_cls[i:i + 1]
        sim = F.cosine_similarity(
            q.unsqueeze(1), ref.unsqueeze(0), dim=2).squeeze(0)

        if agg == 'mean':
            scores[i] = 1.0 - sim.mean().item()
        elif agg == 'topk':
            k = min(3, len(sim))
            scores[i] = 1.0 - sim.topk(k).values.mean().item()
        else:
            scores[i] = 1.0 - sim.max().item()

    median_val = np.nanmedian(scores)
    scores[np.isnan(scores)] = median_val

    if n_miss > 0:
        print(f'  CLS anomaly ({agg}): {n_miss}/{n} queries have no '
              f'reference (filled with median={median_val:.4f})')
    return scores


def score_ref_calibrated(query_probs, ref_prob_bank, query_items):
    """Reference-calibrated classifier probability.

    Idea: if the classifier thinks the reference images are also
    "defective", the item's normal appearance probably just looks
    defect-like.  Subtract a fraction of the reference defect
    probability to reduce false positives.

    Returns calibrated scores (not clamped — let rank normalization handle).
    """
    n = len(query_items)
    ref_adj = np.zeros(n)

    for i, iid in enumerate(query_items):
        if iid in ref_prob_bank:
            ref_adj[i] = ref_prob_bank[iid].mean()

    calibrated = query_probs - ref_adj
    return calibrated


# ------------------------------------------------------------------
# Fusion
# ------------------------------------------------------------------
def rank_normalize(x):
    return rankdata(x) / len(x)


def fuse_rank(classifier_probs, anomaly_scores, alpha):
    """Rank-based fusion: (1-alpha)*rank(cls) + alpha*rank(anomaly)."""
    r_cls = rank_normalize(classifier_probs)
    r_ano = rank_normalize(anomaly_scores)
    return (1.0 - alpha) * r_cls + alpha * r_ano


# ------------------------------------------------------------------
# Evaluation (when labels available)
# ------------------------------------------------------------------
def compute_ap(labels, scores):
    from sklearn.metrics import average_precision_score
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.0
    return average_precision_score(labels, scores) * 100


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Reference-Enhanced Inference for Kaputt')
    p.add_argument('--config', required=True, help='Model config file')
    p.add_argument('--checkpoint', required=True, help='Model checkpoint')
    p.add_argument('--kaputt-root', required=True,
                   help='Dataset root (e.g. /data/public/dataset/kaputt2). '
                        'Must contain query-test.parquet, '
                        'reference-test.parquet, and data/ subdirectory.')
    p.add_argument('--split', default='test',
                   choices=['train', 'validation', 'test'])
    p.add_argument('--output-dir', default='results_ref_enhanced')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--prefetch-factor', type=int, default=8,
                   help='Batches each DataLoader worker prefetches '
                        '(higher = more CPU memory, less GPU starvation)')

    p.add_argument('--tta', action='store_true',
                   help='Use TTA for classifier probabilities')
    p.add_argument('--tta-scales', nargs='+', type=int, default=None,
                   help='Explicit TTA scales in pixels')
    p.add_argument('--multi-gpu', action='store_true',
                   help='Distribute TTA across GPUs')
    p.add_argument('--no-flip', action='store_true',
                   help='Disable flip augmentation in TTA')
    p.add_argument('--no-rotation', action='store_true',
                   help='Disable 90/180/270 rotation augmentation in TTA')
    p.add_argument('--no-diagonal', action='store_true',
                   help='Disable transpose/transverse diagonal flip in TTA')
    p.add_argument('--shift', action='store_true',
                   help='Enable sub-patch pixel shift TTA '
                        '(shifts patch grid by half patch_size)')
    p.add_argument('--shift-pixels', nargs='+', type=int, default=None,
                   help='Shift amounts in pixels (default: patch_size//2). '
                        'Each value generates 4 views: +x, +y, +xy, -xy')
    p.add_argument('--crop', action='store_true',
                   help='Enable center-crop zoom-in TTA')
    p.add_argument('--crop-ratios', nargs='+', type=float, default=None,
                   help='Crop ratios for center-crop TTA '
                        '(default: 0.875 0.95). Smaller = more zoom-in')
    p.add_argument('--five-crop', action='store_true',
                   help='Enable five-crop TTA (4 corners + center)')
    p.add_argument('--tta-weighted', action='store_true',
                   help='Gaussian-weighted TTA averaging')
    p.add_argument('--tta-center', type=int, default=None,
                   help='Center scale for Gaussian weighting (pixels)')
    p.add_argument('--tta-sigma', type=float, default=0.3,
                   help='Sigma for Gaussian scale weighting')

    p.add_argument('--alphas', nargs='+', type=float,
                   default=[0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15,
                            0.20, 0.25, 0.30],
                   help='Fusion weights for anomaly signal')
    p.add_argument('--ref-cal-betas', nargs='+', type=float,
                   default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
                   help='Beta values for reference-calibrated scoring')
    p.add_argument('--anomaly-agg', default='max',
                   choices=['max', 'mean', 'topk'],
                   help='How to aggregate reference similarities: '
                        'max (default), mean, or topk (mean of top-3)')
    p.add_argument('--ref-scales', nargs='+', type=int, default=None,
                   help='Multi-scale sizes for reference feature extraction '
                        '(e.g. 720 880 1024). Averages features across scales '
                        'for more robust matching.')
    return p.parse_args()


def main():
    global _PREFETCH_FACTOR
    args = parse_args()
    _PREFETCH_FACTOR = args.prefetch_factor

    # fork is required so PreloadedDataset's PIL images are shared via COW
    # (not copied) across DataLoader workers.
    torch.multiprocessing.set_start_method('fork', force=True)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.perf_counter()

    root = Path(args.kaputt_root)
    split = args.split

    query_parquet = root / f'query-{split}.parquet'
    ref_parquet = root / f'reference-{split}.parquet'
    query_img_dir = root / 'data' / split / 'query-data' / 'crop'
    ref_img_dir = root / 'data' / split / 'reference-data' / 'crop'

    print('=' * 70)
    print('  Reference-Enhanced Inference')
    print('=' * 70)
    print(f'  Config     : {args.config}')
    print(f'  Checkpoint : {args.checkpoint}')
    print(f'  Kaputt root: {root}')
    print(f'  Split      : {split}')
    print(f'  Query imgs : {query_img_dir}')
    print(f'  Ref imgs   : {ref_img_dir}')
    print(f'  DataLoader : workers={args.num_workers}, '
          f'prefetch={_PREFETCH_FACTOR}')

    # ── Load query data ───────────────────────────────────────────────
    if not query_parquet.exists():
        print(f'\nERROR: {query_parquet} not found.')
        print('The query parquet is needed for item_identifier matching.')
        sys.exit(1)

    query_df = pd.read_parquet(query_parquet)
    capture_ids = query_df['capture_id'].tolist()
    has_labels = 'defect' in query_df.columns
    labels = None
    if has_labels:
        labels = query_df['defect'].astype(int).values

    if 'item_identifier' not in query_df.columns:
        print('\nERROR: query parquet has no item_identifier column.')
        print('Cannot match queries to references without item_identifier.')
        sys.exit(1)

    query_items = query_df['item_identifier'].tolist()
    n_query = len(capture_ids)
    print(f'  Queries    : {n_query}')
    if has_labels:
        print(f'  Labels     : YES (defective={labels.sum()}, '
              f'non-defective={n_query - labels.sum()})')
    else:
        print(f'  Labels     : NO (prediction mode)')

    # ── Build model ───────────────────────────────────────────────────
    print('\nBuilding model ...', flush=True)
    model, cfg, img_size, patch_size = build_model(
        args.config, args.checkpoint, device)
    print(f'  img_size={img_size}, patch_size={patch_size}', flush=True)

    # ── Load & build reference bank ───────────────────────────────────
    print('\nBuilding reference feature bank ...', flush=True)
    if not ref_parquet.exists():
        print(f'  WARNING: {ref_parquet} not found. '
              'Running without reference signals.')
        ref_cls_bank, ref_prob_bank = {}, {}
    else:
        ref_df = pd.read_parquet(ref_parquet)
        print(f'  Reference parquet: {len(ref_df)} rows', flush=True)
        ref_pairs = resolve_ref_paths(ref_df, ref_img_dir, args.kaputt_root)
        print(f'  Resolved {len(ref_pairs)} reference image paths', flush=True)
        ref_cls_bank, ref_prob_bank = build_reference_bank(
            model, ref_pairs, img_size, device,
            batch_size=args.batch_size, num_workers=args.num_workers,
            multi_scale_sizes=args.ref_scales)

    # ── Preload query images into RAM (one-time disk read) ─────────────
    print('\nPreloading query images ...', flush=True)
    query_pil_images = preload_images_by_capture_id(
        capture_ids, query_img_dir, num_workers=args.num_workers)

    # ── Extract query features + base classifier probs ────────────────
    print('\nExtracting query features ...', flush=True)
    tfm_native = get_transform(img_size)
    q_ds = PreloadedDataset(query_pil_images, tfm_native)
    q_ld = _make_loader(q_ds, args.batch_size, args.num_workers)
    query_cls, query_probs_base = extract_features_and_probs(
        model, q_ld, device)

    # ── Classifier probs (with optional TTA) ──────────────────────────
    if args.tta:
        use_flip = not args.no_flip
        use_rotation = not args.no_rotation
        use_diagonal = not args.no_diagonal
        use_multigpu = args.multi_gpu and torch.cuda.device_count() > 1
        tta_opts = (f'flip={use_flip}, rotation={use_rotation}, '
                    f'diagonal={use_diagonal}, shift={args.shift}, '
                    f'crop={args.crop}, five_crop={args.five_crop}, '
                    f'weighted={args.tta_weighted}, '
                    f'multi_gpu={use_multigpu}')
        print(f'\nRunning TTA ({tta_opts}) ...', flush=True)

        def ds_factory(tfm):
            return PreloadedDataset(query_pil_images, tfm)

        tta_kwargs = dict(
            batch_size=args.batch_size, num_workers=args.num_workers,
            explicit_scales=args.tta_scales,
            use_flip=use_flip,
            use_rotation=use_rotation,
            use_diagonal=use_diagonal,
            use_shift=args.shift,
            shift_pixels=args.shift_pixels,
            use_crop=args.crop,
            crop_ratios=args.crop_ratios,
            use_five_crop=args.five_crop,
            weighted=args.tta_weighted,
            tta_center=args.tta_center,
            tta_sigma=args.tta_sigma,
        )

        if use_multigpu:
            classifier_probs = tta_inference_multigpu(
                args.config, args.checkpoint,
                ds_factory, img_size, patch_size,
                **tta_kwargs)
        else:
            classifier_probs = tta_inference(
                model, ds_factory, img_size, patch_size, device,
                **tta_kwargs)
    else:
        classifier_probs = query_probs_base

    # ── Compute anomaly scores ────────────────────────────────────────
    print('\nComputing anomaly scores ...', flush=True)

    cls_anomaly = score_cls_anomaly(
        query_cls, ref_cls_bank, query_items, agg=args.anomaly_agg)
    print(f'  CLS anomaly ({args.anomaly_agg}): '
          f'mean={cls_anomaly.mean():.4f}, std={cls_anomaly.std():.4f}, '
          f'min={cls_anomaly.min():.4f}, max={cls_anomaly.max():.4f}')

    ref_calibrated_raw = score_ref_calibrated(
        classifier_probs, ref_prob_bank, query_items)
    print(f'  Ref-calibrated: mean={ref_calibrated_raw.mean():.4f}, '
          f'std={ref_calibrated_raw.std():.4f}')

    print(f'  Classifier prob: mean={classifier_probs.mean():.4f}, '
          f'std={classifier_probs.std():.4f}')

    # ── Save analysis CSV ─────────────────────────────────────────────
    analysis = pd.DataFrame({
        'capture_id': capture_ids,
        'item_identifier': query_items,
        'classifier_prob': classifier_probs,
        'cls_anomaly': cls_anomaly,
        'ref_calibrated': ref_calibrated_raw,
    })
    if has_labels:
        analysis['label'] = labels
    analysis.to_csv(
        os.path.join(args.output_dir, 'analysis_all_scores.csv'), index=False)

    # ── Generate fused predictions ────────────────────────────────────
    print('\n' + '=' * 70)
    print('  Generating fused predictions')
    print('=' * 70)

    results = {}

    # Baseline: pure classifier
    name = 'classifier_only'
    results[name] = classifier_probs
    _save_pred(args.output_dir, name, capture_ids, classifier_probs)

    # Strategy A: CLS-anomaly rank fusion
    for alpha in args.alphas:
        name = f'cls_anomaly_alpha{alpha:.2f}'
        fused = fuse_rank(classifier_probs, cls_anomaly, alpha)
        results[name] = fused
        _save_pred(args.output_dir, name, capture_ids, fused)

    # Strategy B: Reference-calibrated probability
    for beta in args.ref_cal_betas:
        ref_adj = np.zeros(n_query)
        for i, iid in enumerate(query_items):
            if iid in ref_prob_bank:
                ref_adj[i] = ref_prob_bank[iid].mean()
        calibrated = classifier_probs - beta * ref_adj
        name = f'ref_calibrated_beta{beta:.1f}'
        results[name] = calibrated
        _save_pred(args.output_dir, name, capture_ids, calibrated)

        # Also rank-fused version
        name_rank = f'ref_calibrated_beta{beta:.1f}_rank'
        fused = fuse_rank(classifier_probs, calibrated, 0.5)
        results[name_rank] = fused
        _save_pred(args.output_dir, name_rank, capture_ids, fused)

    # Strategy C: Combined CLS anomaly + ref-calibrated
    for alpha in [0.05, 0.10, 0.15]:
        combined = 0.5 * rank_normalize(cls_anomaly) + \
                   0.5 * rank_normalize(-ref_calibrated_raw)
        name = f'combined_alpha{alpha:.2f}'
        fused = fuse_rank(classifier_probs, combined, alpha)
        results[name] = fused
        _save_pred(args.output_dir, name, capture_ids, fused)

    # Strategy D: Multiplicative boost for high-anomaly samples
    for gamma in [0.1, 0.2, 0.3]:
        cls_rank = rank_normalize(classifier_probs)
        ano_rank = rank_normalize(cls_anomaly)
        boosted = cls_rank * (1.0 + gamma * (ano_rank - 0.5))
        name = f'multiplicative_gamma{gamma:.1f}'
        results[name] = boosted
        _save_pred(args.output_dir, name, capture_ids, boosted)

    # Strategy E: Probability-space anomaly adjustment (not rank-based)
    # The key insight: rank fusion discards magnitude information.
    # In probability space, only adjust samples where anomaly is extreme.
    ano_z = (cls_anomaly - cls_anomaly.mean()) / (cls_anomaly.std() + 1e-8)
    for gamma in [0.02, 0.05, 0.10, 0.15]:
        adjusted = classifier_probs.copy()
        boost_mask = ano_z > 0  # above-average anomaly -> more defective
        adjusted[boost_mask] += gamma * ano_z[boost_mask] * adjusted[boost_mask]
        adjusted = np.clip(adjusted, 0, 1)
        name = f'prob_boost_gamma{gamma:.2f}'
        results[name] = adjusted
        _save_pred(args.output_dir, name, capture_ids, adjusted)

    # Strategy F: Logit-space fusion — more natural for probabilities
    eps = 1e-7
    cls_logit = np.log(np.clip(classifier_probs, eps, 1 - eps) /
                       np.clip(1 - classifier_probs, eps, 1 - eps))
    ano_logit = np.log(np.clip(rank_normalize(cls_anomaly), eps, 1 - eps) /
                       np.clip(1 - rank_normalize(cls_anomaly), eps, 1 - eps))
    for alpha in [0.02, 0.05, 0.10]:
        fused_logit = cls_logit + alpha * ano_logit
        fused_prob = 1.0 / (1.0 + np.exp(-fused_logit))
        name = f'logit_fusion_alpha{alpha:.2f}'
        results[name] = fused_prob
        _save_pred(args.output_dir, name, capture_ids, fused_prob)

    # Strategy G: Selective ref-calibration — only subtract when ref
    # prob is high (reference looks defective -> likely FP)
    for beta in [0.3, 0.5, 0.7, 1.0]:
        ref_adj = np.zeros(n_query)
        for i, iid in enumerate(query_items):
            if iid in ref_prob_bank:
                rp = ref_prob_bank[iid].mean()
                if rp > 0.3:
                    ref_adj[i] = rp
        calibrated = classifier_probs - beta * ref_adj
        name = f'selective_refcal_beta{beta:.1f}'
        results[name] = calibrated
        _save_pred(args.output_dir, name, capture_ids, calibrated)

    # Strategy H: Agreement-weighted — trust classifier more when
    # anomaly score agrees with classifier direction
    cls_rank = rank_normalize(classifier_probs)
    ano_rank = rank_normalize(cls_anomaly)
    agreement = 1.0 - np.abs(cls_rank - ano_rank)
    for alpha in [0.05, 0.10, 0.15]:
        weighted = cls_rank + alpha * agreement * (ano_rank - 0.5)
        name = f'agreement_alpha{alpha:.2f}'
        results[name] = weighted
        _save_pred(args.output_dir, name, capture_ids, weighted)

    # ── Print results ─────────────────────────────────────────────────
    print(f'\n{"Strategy":45s} | {"mean":>7s} | {"std":>7s}', end='')
    if has_labels:
        print(f' | {"AP":>7s}', end='')
    print()
    print('-' * (62 + (10 if has_labels else 0)))

    for name, preds in results.items():
        line = f'{name:45s} | {preds.mean():.4f} | {preds.std():.4f}'
        if has_labels:
            ap = compute_ap(labels, preds)
            line += f' | {ap:6.2f}%'
        print(line)

    elapsed = time.perf_counter() - t0
    print(f'\nTotal time: {elapsed:.1f}s')
    print(f'All results saved to: {args.output_dir}')

    if has_labels:
        print("""
======================================================================
  How to use these results:
======================================================================
  1. Look at the AP column above to find which strategy/alpha improves
     over "classifier_only".
  2. Use that same strategy on Kaputt2 (--kaputt-root .../kaputt2).
  3. If cls_anomaly_alpha0.05 helps on Kaputt1, try it on Kaputt2.
  4. If ref_calibrated_beta0.5 helps, the reference "defect look" signal
     is useful — some items just look defective but aren't.
""")
    else:
        print("""
======================================================================
  Recommendations for submission:
======================================================================
  Submit in this order (most conservative first):
  1. cls_anomaly_alpha0.05   — tiny reference weight, safest
  2. cls_anomaly_alpha0.10   — moderate reference weight
  3. ref_calibrated_beta0.5  — if the item's references also score high,
                               reduce the query's defect score
  4. combined_alpha0.05      — both CLS distance + reference calibration

  IMPORTANT: first validate on Kaputt1 test (with labels) to find the
  best alpha/beta, then apply the same settings to Kaputt2.
""")


def _save_pred(output_dir, name, capture_ids, scores):
    pd.DataFrame({
        'capture_id': capture_ids,
        'pred': scores,
    }).to_csv(os.path.join(output_dir, f'pred_{name}.csv'), index=False)


if __name__ == '__main__':
    main()
