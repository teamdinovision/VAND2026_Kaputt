"""Filter mislabeled samples using trained model confidence (multi-GPU).

Loads a trained checkpoint, runs inference on train/val/test splits
using all visible GPUs in parallel, and removes samples where the
model prediction strongly disagrees with the ground-truth label:

  - Label=defect   but P(defect) < thresh_down  → likely mislabeled
  - Label=no-defect but P(defect) > thresh_up    → likely mislabeled

Outputs cleaned parquet files:
  query-train-2
  query-validation-2
  query-test-2

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python clean_mislabel.py \
        --config configs/large_728_mlp_aux0.3.py \
        --checkpoint work_dirs/large_728_mlp_aux0.3/epoch_10.pth \
        --thresh-up 0.6 --thresh-down 0.5
"""

import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmpretrain_kaputt  # noqa: F401

from mmengine.config import Config
from mmpretrain.registry import MODELS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

SPLITS = [
    ('train',      'query-train.parquet',      'data/train/query-data/crop'),
    ('validation', 'query-validation.parquet',  'data/validation/query-data/crop'),
    ('test',       'query-test.parquet',        'data/test/query-data/crop'),
]


class SimpleDataset(Dataset):
    def __init__(self, df, image_root, transform):
        self.capture_ids = df['capture_id'].values
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.capture_ids)

    def __getitem__(self, idx):
        cid = self.capture_ids[idx]
        img = Image.open(
            self.image_root / f'{cid}.jpg').convert('RGB')
        img = self.transform(img)
        return img


def _build_cfg(config_path):
    cfg = Config.fromfile(config_path)
    cfg.model.backbone.pretrained = False
    model_name = cfg.model.backbone.get('model_name', '')
    if 'dinov2' in model_name:
        cfg.model.backbone.dynamic_img_size = True
    return cfg


def build_model_on_device(cfg, checkpoint_path, device):
    model = MODELS.build(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def _infer_shard(model, loader, device, gpu_id, quiet=False):
    probs_list = []
    for images in tqdm(loader, desc=f'  GPU{gpu_id}', leave=False,
                       disable=quiet):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(images, mode='tensor')
        probs = F.softmax(logits.float(), dim=1)[:, 1]
        probs_list.append(probs.cpu())
    if not probs_list:
        return np.array([], dtype=np.float32)
    return torch.cat(probs_list).numpy()


def run_split_multigpu(models, img_size, df, image_root,
                       batch_size, num_workers):
    """Run inference across all GPUs, each processing a data shard."""
    transform = transforms.Compose([
        transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    full_ds = SimpleDataset(df, image_root, transform)

    num_gpus = len(models)
    n = len(full_ds)
    shard_size = (n + num_gpus - 1) // num_gpus
    workers_per_gpu = max(2, num_workers // num_gpus)

    shard_results = [None] * num_gpus
    shard_indices = []
    for gid in range(num_gpus):
        start = gid * shard_size
        end = min(start + shard_size, n)
        shard_indices.append((start, end))

    def _gpu_worker(gid):
        model, device = models[gid]
        start, end = shard_indices[gid]
        if start >= end:
            shard_results[gid] = np.array([], dtype=np.float32)
            return
        subset = Subset(full_ds, list(range(start, end)))
        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=False,
            num_workers=workers_per_gpu, pin_memory=True,
            prefetch_factor=4 if workers_per_gpu > 0 else None,
            persistent_workers=workers_per_gpu > 0,
            drop_last=False,
        )
        shard_results[gid] = _infer_shard(model, loader, device, gid)

    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futs = [pool.submit(_gpu_worker, gid) for gid in range(num_gpus)]
        for f in futs:
            f.result()

    return np.concatenate(shard_results)


def filter_mislabeled(df, probs, thresh_up, thresh_down):
    labels = df['defect'].values.astype(int)

    is_defect = labels == 1
    is_normal = labels == 0

    bad_defect = is_defect & (probs < thresh_down)
    bad_normal = is_normal & (probs > thresh_up)
    mislabeled = bad_defect | bad_normal

    return mislabeled, bad_defect, bad_normal


def print_stats(split_name, df, probs, mislabeled, bad_defect, bad_normal,
                thresh_up, thresh_down):
    labels = df['defect'].values.astype(int)
    total = len(df)
    n_defect = int(labels.sum())
    n_normal = total - n_defect

    n_bad_defect = int(bad_defect.sum())
    n_bad_normal = int(bad_normal.sum())
    n_mislabeled = int(mislabeled.sum())

    print(f'\n{"=" * 70}')
    print(f'  [{split_name}]  共 {total} 样本  '
          f'(缺陷={n_defect}, 正常={n_normal})')
    print(f'{"=" * 70}')

    print(f'  置信度统计:')
    print(f'    全部样本   mean={probs.mean():.4f}  '
          f'std={probs.std():.4f}  '
          f'min={probs.min():.4f}  max={probs.max():.4f}')
    if n_defect > 0:
        dp = probs[labels == 1]
        print(f'    缺陷样本   mean={dp.mean():.4f}  '
              f'std={dp.std():.4f}  '
              f'min={dp.min():.4f}  max={dp.max():.4f}')
    if n_normal > 0:
        np_ = probs[labels == 0]
        print(f'    正常样本   mean={np_.mean():.4f}  '
              f'std={np_.std():.4f}  '
              f'min={np_.min():.4f}  max={np_.max():.4f}')

    print(f'  错标检测 (thresh_up={thresh_up}, thresh_down={thresh_down}):')
    print(f'    标为缺陷但置信度 < {thresh_down}: '
          f'{n_bad_defect}/{n_defect} '
          f'({n_bad_defect/max(n_defect,1)*100:.1f}%)')
    print(f'    标为正常但置信度 > {thresh_up}: '
          f'{n_bad_normal}/{n_normal} '
          f'({n_bad_normal/max(n_normal,1)*100:.1f}%)')
    print(f'    总共错标: {n_mislabeled}/{total} '
          f'({n_mislabeled/total*100:.1f}%)')
    print(f'    清洗后保留: {total - n_mislabeled}/{total} '
          f'({(total - n_mislabeled)/total*100:.1f}%)')

    if 'item_material' in df.columns:
        print(f'  按材质分布:')
        for mat in sorted(df['item_material'].dropna().unique()):
            mask = df['item_material'].values == mat
            mat_total = int(mask.sum())
            mat_bad = int(mislabeled[mask].sum())
            if mat_bad > 0:
                print(f'    {mat:30s}  错标 {mat_bad}/{mat_total} '
                      f'({mat_bad/mat_total*100:.1f}%)')

    if n_mislabeled > 0:
        bad_idx = np.where(mislabeled)[0]
        print(f'  错标样本 (前20个):')
        for i, idx in enumerate(bad_idx[:20]):
            row = df.iloc[idx]
            tag = '缺陷→低' if bad_defect[idx] else '正常→高'
            mat = getattr(row, 'item_material', '?')
            print(f'    [{tag}] capture_id={row.capture_id}  '
                  f'label={int(row.defect)}  prob={probs[idx]:.4f}  '
                  f'material={mat}')
        if len(bad_idx) > 20:
            print(f'    ... 还有 {len(bad_idx) - 20} 个')


def main():
    parser = argparse.ArgumentParser(
        description='Filter mislabeled samples using model confidence')
    parser.add_argument('--config', type=str,
                        default='configs/large_728_mlp_aux0.3.py')
    parser.add_argument('--checkpoint', type=str,
                        default='work_dirs/large_728_mlp_aux0.3/epoch_10.pth')
    parser.add_argument('--thresh-up', type=float, default=0.6,
                        help='Normal samples with prob > thresh_up are suspect')
    parser.add_argument('--thresh-down', type=float, default=0.5,
                        help='Defect samples with prob < thresh_down are suspect')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--parquet-root', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Directory to save cleaned parquet files')
    args = parser.parse_args()

    cfg = _build_cfg(args.config)
    data_root = args.data_root or cfg.get('data_root', '/data/public/dataset/kaputt')
    parquet_root = args.parquet_root or cfg.get('parquet_root', '/data/public/dataset/kaputt')
    img_size = cfg.get('img_size', 518)

    num_gpus = torch.cuda.device_count()
    print(f'GPUs: {num_gpus}')
    print(f'Config: {args.config}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Thresholds: up={args.thresh_up}, down={args.thresh_down}')
    print(f'Data root: {data_root}')
    print(f'Parquet root: {parquet_root}')

    models = []
    for gid in range(max(num_gpus, 1)):
        device = torch.device(f'cuda:{gid}' if num_gpus > 0 else 'cpu')
        m = build_model_on_device(cfg, args.checkpoint, device)
        models.append((m, device))
    print(f'Models loaded on {len(models)} device(s), img_size={img_size}')

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    total_removed = 0
    total_samples = 0

    for split_name, parquet_file, img_subdir in SPLITS:
        parquet_path = os.path.join(parquet_root, parquet_file)
        image_root = os.path.join(data_root, img_subdir)

        if not os.path.exists(parquet_path):
            print(f'\n[SKIP] {parquet_path} 不存在')
            continue

        df = pd.read_parquet(parquet_path)
        print(f'\n推理 [{split_name}] ... ({len(df)} 样本)')

        probs = run_split_multigpu(
            models, img_size, df, image_root,
            args.batch_size, args.num_workers)

        mislabeled, bad_defect, bad_normal = filter_mislabeled(
            df, probs, args.thresh_up, args.thresh_down)

        print_stats(split_name, df, probs, mislabeled, bad_defect, bad_normal,
                    args.thresh_up, args.thresh_down)

        clean_df = df[~mislabeled].reset_index(drop=True)
        clean_name = parquet_file.replace('.parquet', '-clean2.parquet')
        clean_path = os.path.join(output_dir, clean_name)
        clean_df.to_parquet(clean_path, index=False)
        print(f'  保存: {clean_path} ({len(clean_df)} 样本)')

        total_removed += int(mislabeled.sum())
        total_samples += len(df)

    print(f'\n{"=" * 70}')
    print(f'  汇总: 共 {total_samples} 样本, '
          f'剔除 {total_removed} 个错标 '
          f'({total_removed/max(total_samples,1)*100:.1f}%), '
          f'保留 {total_samples - total_removed} 个')
    print(f'{"=" * 70}')
    print('Done.')


if __name__ == '__main__':
    main()
