"""
Kaputt Defect Detection — MMPretrain DINOv2 / ConvNeXt Training
================================================================
Wraps MMEngine Runner for single / multi-GPU training.

Usage:
    # Single GPU — DINOv2-L
    python train_vit_mm.py configs/kaputt_dinov2.py

    # Single GPU — ConvNeXt-V2 (for ensemble)
    python train_vit_mm.py configs/kaputt_convnext.py

    # 4× A100 (torchrun / DDP)
    torchrun --nproc_per_node=4 train_vit_mm.py configs/kaputt_dinov2.py

    # Override any config value on the fly
    python train_vit_mm.py configs/kaputt_dinov2.py \\
        --cfg-options backbone_name=dinov2_b \\
                      train_dataloader.batch_size=16

    # Resume from latest checkpoint
    python train_vit_mm.py configs/kaputt_dinov2.py --resume
"""

import os
import sys
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmpretrain_kaputt  # noqa: F401  — registers custom modules

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    def record(fn):
        return fn


def parse_args():
    p = argparse.ArgumentParser(description='Kaputt MMPretrain Training')
    p.add_argument('config', help='Path to config file')
    p.add_argument('--work-dir', default=None, help='Override work directory')
    p.add_argument('--resume', action='store_true',
                   help='Resume training from the latest checkpoint')
    p.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='Override config values, e.g. --cfg-options epochs=50')
    return p.parse_args()


@record
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.resume:
        cfg.resume = True

    cfg.launcher = 'pytorch' if 'LOCAL_RANK' in os.environ else 'none'

    runner = Runner.from_cfg(cfg)
    runner.train()

    # After training, remind about SWA model if SWAHook was used
    swa_path = os.path.join(cfg.work_dir, 'swa_model.pth')
    if os.path.exists(swa_path):
        print(f'\n{"=" * 64}')
        print(f'SWA model available: {swa_path}')
        print('Use it as an extra checkpoint for ensemble evaluation:')
        print(f'  python evaluate_vit_mm.py --config {args.config} \\')
        print(f'      --checkpoint best.pth {swa_path} --ensemble --tta')
        print(f'{"=" * 64}\n')


if __name__ == '__main__':
    main()
