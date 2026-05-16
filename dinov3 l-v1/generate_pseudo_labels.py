"""Pseudo-label Kaputt reference images using a trained defect classifier.

Reference images are nominally "non-defective" but the paper notes that
only ~20% are truly normal.  This script uses a trained model to assign
confidence-scored pseudo-labels so they can be safely incorporated into
training.

Usage:
    # Basic — single-scale inference
    python generate_pseudo_labels.py \
        --config configs/large_728_three_freeze.py \
        --checkpoint work_dirs/kaputt_dinov3_l_mm/best.pth \
        --data-root /data/public/dataset/kaputt \
        --parquet-root /data/public/dataset/kaputt \
        --output pseudo_labels_ref.csv

    # TTA — multi-scale for more robust pseudo-labels (recommended)
    python generate_pseudo_labels.py \
        --config configs/large_728_three_freeze.py \
        --checkpoint work_dirs/kaputt_dinov3_l_mm/best.pth \
        --data-root /data/public/dataset/kaputt \
        --parquet-root /data/public/dataset/kaputt \
        --output pseudo_labels_ref.csv \
        --tta-scales 512 728 960

Recommended workflow:
    1. Train Phase-1 model on query data only (existing pipeline)
    2. Run this script to generate pseudo-labels for reference images
    3. Train Phase-2 model with configs/large_728_ref.py (loads Phase-1 ckpt)
    4. (Optional) Re-run this script with Phase-2 model for iterative refinement
"""

import os
import sys
import argparse
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
from mmpretrain.registry import MODELS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RefImageDataset(Dataset):
    """Lightweight dataset wrapping a list of image file paths."""

    def __init__(self, image_paths, transform):
        self.paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), idx


def _build_model(cfg_path, ckpt_path, device):
    cfg = Config.fromfile(cfg_path)
    cfg.model.backbone.pretrained = False
    name = cfg.model.backbone.get('model_name', '')
    if 'dinov2' in name or 'dinov3' in name:
        cfg.model.backbone.dynamic_img_size = True
    model = MODELS.build(cfg.model)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(sd.get('state_dict', sd), strict=False)
    return model.to(device).eval(), cfg.get('img_size', 518)


def _make_transform(size):
    return transforms.Compose([
        transforms.Resize(
            (size, size),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def _infer(model, loader, device):
    out = []
    for imgs, _ in tqdm(loader, desc='  infer', leave=False):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(imgs, mode='tensor')
        out.append(F.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def _collect_ref_images(parquet_root, data_root, subset):
    """Parse reference parquet and return (image_paths, item_identifiers)."""
    pq = os.path.join(parquet_root, f'reference-{subset}.parquet')
    if not os.path.exists(pq):
        return [], []

    df = pd.read_parquet(pq)
    crop_dir = os.path.join(
        data_root, 'data', subset, 'reference-data', 'crop')

    paths, item_ids = [], []
    missing = 0
    for _, row in df.iterrows():
        crops = str(row.reference_crop).split(',')
        for c in crops:
            c = c.strip()
            if not c:
                continue
            fname = os.path.basename(c)
            full = os.path.join(crop_dir, fname)
            if not os.path.exists(full):
                alt = os.path.join(parquet_root, c)
                if os.path.exists(alt):
                    full = alt
                else:
                    missing += 1
                    continue
            paths.append(full)
            item_ids.append(row.item_identifier)

    if missing:
        print(f'    WARNING: {missing} reference crops not found in {subset}')
    return paths, item_ids


def parse_args():
    p = argparse.ArgumentParser(
        description='Pseudo-label Kaputt reference images')
    p.add_argument('--config', required=True,
                   help='Config file of the trained model')
    p.add_argument('--checkpoint', required=True,
                   help='Checkpoint of the trained model')
    p.add_argument('--data-root', required=True,
                   help='Root directory of the Kaputt image data')
    p.add_argument('--parquet-root', required=True,
                   help='Directory containing reference-*.parquet files')
    p.add_argument('--output', default='pseudo_labels_ref.csv',
                   help='Output CSV path')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--subsets', nargs='+',
                   default=['train', 'validation', 'test'],
                   help='Dataset subsets to process')
    p.add_argument('--tta-scales', nargs='+', type=int, default=None,
                   help='Multi-scale TTA for more robust pseudo-labels')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, base_size = _build_model(args.config, args.checkpoint, device)
    print(f'Model loaded.  Base image size: {base_size}')

    records = []
    for subset in args.subsets:
        paths, item_ids = _collect_ref_images(
            args.parquet_root, args.data_root, subset)
        if not paths:
            print(f'  {subset}: no reference images found, skipping')
            continue
        print(f'  {subset}: {len(paths)} reference crops')

        scales = sorted(set([base_size] + (args.tta_scales or [])))
        all_probs = []
        for s in scales:
            ds = RefImageDataset(paths, _make_transform(s))
            ld = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
            print(f'    scale {s}\u00d7{s}')
            all_probs.append(_infer(model, ld, device))

        probs = np.mean(all_probs, axis=0)

        for i, (p, iid) in enumerate(zip(probs, item_ids)):
            records.append(dict(
                img_path=paths[i],
                item_identifier=iid,
                subset=subset,
                defect_prob=float(p),
            ))

    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)

    n = len(df)
    bins = [
        ('Confident normal   (p<0.15)', df['defect_prob'] < 0.15),
        ('Likely normal       (0.15-0.3)',
         (df['defect_prob'] >= 0.15) & (df['defect_prob'] < 0.3)),
        ('Uncertain           (0.3-0.7)',
         (df['defect_prob'] >= 0.3) & (df['defect_prob'] < 0.7)),
        ('Likely defective    (0.7-0.85)',
         (df['defect_prob'] >= 0.7) & (df['defect_prob'] < 0.85)),
        ('Confident defective (p>0.85)', df['defect_prob'] >= 0.85),
    ]
    print(f'\n{"=" * 64}')
    print(f'  Total reference images: {n}')
    for label, mask in bins:
        cnt = int(mask.sum())
        print(f'  {label}: {cnt:6d}  ({cnt / n * 100:5.1f}%)')
    print(f'{"=" * 64}')
    print(f'  Saved \u2192 {args.output}')


if __name__ == '__main__':
    main()
