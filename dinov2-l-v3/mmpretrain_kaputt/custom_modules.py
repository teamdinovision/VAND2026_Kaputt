"""
Custom modules for Kaputt defect classification with MMPretrain.

Includes:
  - CLSTokenNeck           : CLS token pooling for TIMMBackbone ViT output
  - SoftFocalLoss          : Focal Loss + OHEM, supports soft labels (Mixup/CutMix)
  - AsymmetricLoss         : Asymmetric focal loss for AP-optimal binary classification
  - GradualUnfreezeHook    : Progressive backbone layer unfreezing (optional)
  - SWAHook                : Stochastic Weight Averaging
  - DefectAwareTransform   : Stronger augmentation for defective samples
  - RandomRotation90       : Safe 90-degree rotation augmentation
  - MultiScaleResize       : Multi-scale resize + pad, no crop (recommended)
  - MultiScaleResizeCrop   : Multi-scale resize + random crop (alternative)
  - BinaryAPMetric         : Average Precision metric for binary classification
  - MaterialAPMetric       : AP for a specific item_material subset
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmengine.evaluator import BaseMetric
from mmcv.transforms import BaseTransform
from sklearn.metrics import average_precision_score

from mmpretrain.registry import MODELS, HOOKS, TRANSFORMS, METRICS


# ===================================================================
#  Neck: CLS token pooling for TIMMBackbone ViT output
# ===================================================================

@MODELS.register_module()
class CLSTokenNeck(nn.Module):
    """Extract the CLS token from a ViT sequence output.

    TIMMBackbone returns ``((B, N, D),)`` from ``forward_features()``.
    This neck converts it to ``((B, D),)`` by selecting token index 0
    (the CLS token), making it compatible with ``LinearClsHead``.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        if isinstance(inputs, (tuple, list)):
            x = inputs[-1]
        else:
            x = inputs
        if x.dim() == 3:
            x = x[:, 0]  # CLS token → (B, D)
        return (x,)


# ===================================================================
#  [ABLATION] Trick: Focal Loss with Online Hard Example Mining
#  To use: set head.loss.type='SoftFocalLoss' in config
# ===================================================================

@MODELS.register_module()
class SoftFocalLoss(nn.Module):
    """Focal Loss with OHEM that handles both hard and soft labels.

    Supports Mixup / CutMix (soft labels of shape ``(B, C)``) and standard
    integer labels ``(B,)`` transparently.

    Args:
        gamma (float): Focusing parameter — down-weights easy samples.
        alpha (float): Weight for the *positive* (defective) class.
        ohem_ratio (float): Fraction of hardest samples to keep per batch.
            Set to 1.0 to disable OHEM.
        loss_weight (float): Scalar multiplier applied to the final loss.
    """

    def __init__(self, gamma=2.0, alpha=0.75, ohem_ratio=1.0, loss_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ohem_ratio = ohem_ratio
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, **kwargs):
        num_classes = pred.size(1)

        if target.dim() == 1:
            soft_target = F.one_hot(target.long(), num_classes).float()
        else:
            soft_target = target.float()

        log_prob = F.log_softmax(pred, dim=1)
        prob = log_prob.exp()

        focal_weight = (1.0 - prob).pow(self.gamma)

        alpha_weight = torch.ones_like(soft_target)
        alpha_weight[:, 0] = 1.0 - self.alpha
        if num_classes > 1:
            alpha_weight[:, 1] = self.alpha

        per_sample = -(alpha_weight * focal_weight
                       * soft_target * log_prob).sum(dim=1)

        if self.ohem_ratio < 1.0 and per_sample.numel() > 1:
            k = max(1, int(per_sample.numel() * self.ohem_ratio))
            per_sample, _ = per_sample.topk(k)

        if avg_factor is not None:
            loss = per_sample.sum() / avg_factor
        else:
            loss = per_sample.mean()

        return loss * self.loss_weight


# ===================================================================
#  [ABLATION] Trick: Asymmetric Loss — optimized for AP in imbalanced
#  binary classification (recommended over Focal Loss for AP metric)
#  To use: set head.loss.type='AsymmetricLoss' in config
#  To disable: switch back to SoftFocalLoss
#  Reference: Asymmetric Loss For Multi-Label Classification (ICCV 2021)
# ===================================================================

@MODELS.register_module()
class AsymmetricFocalLoss(nn.Module):
    """Asymmetric focal loss adapted for 2-class softmax classification.

    Uses different focusing parameters for positive (defective) and
    negative (non-defective) samples:
      * gamma_pos=0 : never down-weight hard positives — learn ALL defects
      * gamma_neg=4 : strongly suppress easy negatives — most non-defective
                      samples are trivial
    This asymmetry improves Average Precision by preserving gradients from
    hard positive samples while reducing noise from trivial negatives.

    Also includes label smoothing for better probability calibration,
    which helps AP by improving ranking quality.

    Supports soft labels from Mixup / CutMix.

    Args:
        gamma_neg (float): Focusing parameter for non-defective samples.
        gamma_pos (float): Focusing parameter for defective samples.
        label_smoothing (float): Label smoothing factor (0 = off).
        loss_weight (float): Scalar multiplier for the final loss.
    """

    def __init__(self, gamma_neg=4, gamma_pos=0, label_smoothing=0.05,
                 loss_weight=1.0):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.label_smoothing = label_smoothing
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, **kwargs):
        num_classes = pred.size(1)

        if target.dim() == 1:
            soft_target = F.one_hot(target.long(), num_classes).float()
        else:
            soft_target = target.float()

        # Label smoothing for better probability calibration
        if self.label_smoothing > 0:
            soft_target = (soft_target * (1 - self.label_smoothing)
                           + self.label_smoothing / num_classes)

        log_prob = F.log_softmax(pred, dim=1)
        prob = log_prob.exp()

        # Asymmetric gamma: different focusing for positives vs negatives
        is_positive = (soft_target[:, 1] > 0.5).float().unsqueeze(1)  # (B,1)
        gamma = (is_positive * self.gamma_pos
                 + (1 - is_positive) * self.gamma_neg)

        focal_weight = (1.0 - prob).pow(gamma)

        per_sample = -(focal_weight * soft_target * log_prob).sum(dim=1)

        if avg_factor is not None:
            loss = per_sample.sum() / avg_factor
        else:
            loss = per_sample.mean()

        return loss * self.loss_weight


# ===================================================================
#  [ABLATION] Trick: Noise-Robust Focal Loss (Symmetric Cross-Entropy)
#  Combines standard Focal Loss with a reverse cross-entropy (RCE) term.
#  RCE is bounded and less sensitive to noisy labels, preventing the
#  model from memorising incorrect pseudo-labels.
#  Reference: Wang et al., "Symmetric Cross Entropy for Robust Learning
#  with Noisy Labels", ICCV 2019.
#  To use: set head.loss.type='NoisyRobustFocalLoss' in config
# ===================================================================

@MODELS.register_module()
class NoisyRobustFocalLoss(nn.Module):
    """Focal Loss + Symmetric Cross-Entropy for learning with noisy labels.

    L = FocalLoss(y, p) + beta * RCE(p, y)

    The reverse CE term ``RCE = -sum p(k) * log(y(k) + eps)`` is
    bounded and provides a natural regulariser that discourages the
    model from over-fitting to incorrectly labelled samples.

    Args:
        gamma (float): Focal-loss focusing parameter.
        alpha (float): Weight for the positive (defective) class.
        beta (float): Weight of the reverse-CE regulariser.
            0 = standard focal loss, >0 = noise-robust.
        eps (float): Clamp floor for log in the RCE term.
        ohem_ratio (float): Keep this fraction of hardest samples.
        loss_weight (float): Scalar multiplier for the total loss.
    """

    def __init__(self, gamma=2.0, alpha=0.75, beta=0.5, eps=1e-4,
                 ohem_ratio=1.0, loss_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.ohem_ratio = ohem_ratio
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, **kwargs):
        num_classes = pred.size(1)

        if target.dim() == 1:
            soft_target = F.one_hot(target.long(), num_classes).float()
        else:
            soft_target = target.float()

        log_prob = F.log_softmax(pred, dim=1)
        prob = log_prob.exp()

        # --- standard focal-loss term ---
        focal_weight = (1.0 - prob).pow(self.gamma)
        alpha_w = torch.ones_like(soft_target)
        alpha_w[:, 0] = 1.0 - self.alpha
        if num_classes > 1:
            alpha_w[:, 1] = self.alpha

        ce_term = -(alpha_w * focal_weight
                    * soft_target * log_prob).sum(dim=1)

        # --- reverse cross-entropy (noise-robust regulariser) ---
        rce_term = -(prob * torch.log(
            soft_target.clamp(min=self.eps))).sum(dim=1)

        per_sample = ce_term + self.beta * rce_term

        if self.ohem_ratio < 1.0 and per_sample.numel() > 1:
            k = max(1, int(per_sample.numel() * self.ohem_ratio))
            per_sample, _ = per_sample.topk(k)

        if avg_factor is not None:
            loss = per_sample.sum() / avg_factor
        else:
            loss = per_sample.mean()

        return loss * self.loss_weight


# ===================================================================
#  [ABLATION] Trick: Gradual Unfreezing + Gradient Checkpointing
#  To disable: remove GradualUnfreezeHook from custom_hooks in config
#  Note: Current recommended approach is full fine-tune from epoch 1
# ===================================================================

@HOOKS.register_module()
class GradualUnfreezeHook(Hook):
    """Progressively unfreeze backbone transformer blocks.

    Works with ``TIMMBackbone`` (``backbone.timm_model.blocks``) and
    native ``VisionTransformer`` (``backbone.layers``).

    Args:
        unfreeze_schedule (dict[int, int]):
            ``{epoch: num_blocks_to_unfreeze_from_end}``.
            Use ``-1`` to unfreeze everything.
        grad_checkpointing (bool): Enable gradient checkpointing on backbone.
    """

    priority = 'NORMAL'

    def __init__(self, unfreeze_schedule, grad_checkpointing=True):
        self.schedule = {int(k): v for k, v in unfreeze_schedule.items()}
        self.grad_checkpointing = grad_checkpointing

    @staticmethod
    def _unwrap(runner):
        m = runner.model
        return m.module if hasattr(m, 'module') else m

    @staticmethod
    def _get_blocks(model):
        bb = model.backbone
        if hasattr(bb, 'timm_model'):
            tm = bb.timm_model
            if hasattr(tm, 'blocks'):          # ViT / DeiT / Swin
                return list(tm.blocks)
            if hasattr(tm, 'stages'):           # ConvNeXt / ConvNeXtV2
                return list(tm.stages)
        if hasattr(bb, 'layers'):
            return list(bb.layers)
        return []

    def before_run(self, runner):
        if not self.grad_checkpointing:
            return
        model = self._unwrap(runner)
        bb = model.backbone
        if hasattr(bb, 'timm_model') and hasattr(
                bb.timm_model, 'set_grad_checkpointing'):
            bb.timm_model.set_grad_checkpointing(True)
            runner.logger.info(
                'Gradient checkpointing enabled on TIMMBackbone')

    def before_train_epoch(self, runner):
        epoch = runner.epoch + 1
        model = self._unwrap(runner)
        blocks = self._get_blocks(model)
        if not blocks:
            return
        n_total = len(blocks)

        n_unfreeze = 0
        for e in sorted(self.schedule):
            if epoch >= e:
                n_unfreeze = self.schedule[e]

        backbone = model.backbone
        if n_unfreeze == -1:
            for p in backbone.parameters():
                p.requires_grad = True
            runner.logger.info(
                f'Epoch {epoch}: ALL {n_total} backbone blocks unfrozen')
        else:
            for p in backbone.parameters():
                p.requires_grad = False
            for blk in blocks[-n_unfreeze:] if n_unfreeze > 0 else []:
                for p in blk.parameters():
                    p.requires_grad = True
            runner.logger.info(
                f'Epoch {epoch}: last {n_unfreeze}/{n_total} blocks unfrozen')


# ===================================================================
#  [ABLATION] Trick: Gradient Checkpointing Hook (standalone)
#  Enables gradient checkpointing without gradual unfreezing.
#  To disable: remove GradCheckpointHook from custom_hooks in config
# ===================================================================

@HOOKS.register_module()
class GradCheckpointHook(Hook):
    """Enable gradient checkpointing on TIMMBackbone to save GPU memory.

    This is a standalone hook — use instead of GradualUnfreezeHook
    when doing full fine-tuning from epoch 1.
    """

    priority = 'VERY_HIGH'

    def before_run(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        bb = model.backbone
        if hasattr(bb, 'timm_model') and hasattr(
                bb.timm_model, 'set_grad_checkpointing'):
            bb.timm_model.set_grad_checkpointing(True)
            runner.logger.info(
                'GradCheckpointHook: gradient checkpointing enabled')


# ===================================================================
#  [ABLATION] Trick: Stochastic Weight Averaging (SWA)
#  Averages model weights from later epochs for better generalization.
#  To disable: remove SWAHook from custom_hooks in config
#  Output: saves swa_model.pth in work_dir after training
# ===================================================================

@HOOKS.register_module()
class SWAHook(Hook):
    """Stochastic Weight Averaging — uniform average of model weights.

    After ``swa_start_epoch``, accumulates a running average of model
    parameters every ``swa_freq`` epochs. At the end of training, saves
    the averaged model as ``swa_model.pth`` in ``work_dir``.

    Args:
        swa_start_epoch (int): First epoch to start averaging (1-indexed).
        swa_freq (int): Average every N epochs after start.
    """

    priority = 'LOW'

    def __init__(self, swa_start_epoch=25, swa_freq=1):
        self.swa_start_epoch = swa_start_epoch
        self.swa_freq = swa_freq
        self.swa_state = None
        self.n_averaged = 0

    @staticmethod
    def _unwrap(runner):
        m = runner.model
        return m.module if hasattr(m, 'module') else m

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1
        if epoch < self.swa_start_epoch:
            return
        if (epoch - self.swa_start_epoch) % self.swa_freq != 0:
            return

        model = self._unwrap(runner)
        state = model.state_dict()

        if self.swa_state is None:
            self.swa_state = {
                k: v.clone().float() for k, v in state.items()}
            self.n_averaged = 1
        else:
            self.n_averaged += 1
            for k in self.swa_state:
                self.swa_state[k] += (
                    state[k].float() - self.swa_state[k]) / self.n_averaged

        runner.logger.info(
            f'SWA: averaged {self.n_averaged} checkpoints '
            f'(epoch {epoch})')

    def after_run(self, runner):
        if self.swa_state is None:
            runner.logger.warning('SWA: no checkpoints were averaged')
            return

        model = self._unwrap(runner)
        orig_state = model.state_dict()
        swa_state = {}
        for k, v in self.swa_state.items():
            swa_state[k] = v.to(dtype=orig_state[k].dtype)

        save_path = os.path.join(runner.work_dir, 'swa_model.pth')
        torch.save(
            {'state_dict': swa_state,
             'meta': {'n_averaged': self.n_averaged}},
            save_path)
        runner.logger.info(
            f'SWA model saved to {save_path} '
            f'({self.n_averaged} checkpoints averaged)')


# ===================================================================
#  [ABLATION] Trick: Defect-Type-Aware Data Augmentation
#  To disable: remove DefectAwareTransform from train_pipeline
# ===================================================================

@TRANSFORMS.register_module()
class DefectAwareTransform(BaseTransform):
    """Apply extra augmentation exclusively to *defective* samples.

    Randomly selects one of: Gaussian noise, Gaussian blur,
    brightness jitter, or contrast jitter.

    Args:
        extra_prob (float): Probability of applying the extra augmentation.
        noise_sigma (float): Std-dev for Gaussian noise (pixel scale 0-255).
    """

    def __init__(self, extra_prob=0.3, noise_sigma=10.0):
        super().__init__()
        self.extra_prob = extra_prob
        self.noise_sigma = noise_sigma

    def transform(self, results):
        if results.get('gt_label', 0) != 1:
            return results
        if np.random.rand() > self.extra_prob:
            return results

        img = results['img'].copy()
        aug = np.random.choice(['noise', 'blur', 'brightness', 'contrast'])

        if aug == 'noise':
            noise = np.random.normal(0, self.noise_sigma, img.shape)
            img = np.clip(
                img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif aug == 'blur':
            import cv2
            k = int(np.random.choice([3, 5]))
            img = cv2.GaussianBlur(img, (k, k), 0)
        elif aug == 'brightness':
            factor = np.random.uniform(0.85, 1.15)
            img = np.clip(
                img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        else:  # contrast
            mean_val = img.astype(np.float32).mean()
            factor = np.random.uniform(0.85, 1.15)
            img = np.clip(
                (img.astype(np.float32) - mean_val) * factor + mean_val,
                0, 255).astype(np.uint8)

        results['img'] = img
        return results


# ===================================================================
#  [ABLATION] Trick: Safe 90-degree Rotation Augmentation
#  To disable: remove RandomRotation90 from train_pipeline
# ===================================================================

@TRANSFORMS.register_module()
class RandomRotation90(BaseTransform):
    """Rotate image by 0, 90, 180, or 270 degrees.

    Unlike arbitrary-angle rotation, 90-degree multiples are lossless
    (no interpolation artifacts) and safe for defect detection where
    small texture details matter.
    """

    def transform(self, results):
        k = np.random.randint(4)  # 0, 1, 2, or 3
        if k > 0:
            results['img'] = np.rot90(results['img'], k).copy()
        return results


# ===================================================================
#  Multi-scale Resize WITHOUT cropping (recommended for defect cls)
#  Preserves the full image at every scale — no label noise from
#  accidentally cropping away the defect region.
# ===================================================================

@TRANSFORMS.register_module()
class MultiScaleResize(BaseTransform):
    """Resize to a random scale from a fixed list, then pad to a uniform size.

    Every image is always shown in its entirety — no content is lost.
    Padding (with ImageNet-mean pixels) fills the bottom-right margin so
    that all images in a batch have identical spatial dimensions.

    After normalisation the padded region becomes ≈ 0, which self-attention
    naturally down-weights over the course of training.

    Args:
        scales (list[tuple[int,int]]): Candidate (H, W) resize targets.
            Must all be multiples of patch_size and <= ``max_size``.
        max_size (tuple[int,int] | None): Pad to this (H, W).
            Defaults to the largest entry in ``scales``.
        pad_val (tuple[int,int,int]): BGR pixel value used for padding.
            Default ≈ ImageNet mean in BGR so that post-normalisation
            the padding region is close to zero.
        interpolation (str): 'bicubic' | 'bilinear' | 'lanczos' | 'area'.
    """

    _CV2_INTERP = None

    def __init__(self, scales, max_size=None,
                 pad_val=(104, 116, 124), interpolation='bicubic'):
        super().__init__()
        self.scales = [(h, w) if isinstance(h, int) else (h[0], h[1])
                       for h, w in scales]
        if max_size is None:
            mh = max(s[0] for s in self.scales)
            mw = max(s[1] for s in self.scales)
            self.max_size = (mh, mw)
        else:
            self.max_size = tuple(max_size)
        self.pad_val = pad_val
        self._interp_name = interpolation

    def _interp(self):
        if self._CV2_INTERP is None:
            import cv2
            MultiScaleResize._CV2_INTERP = {
                'bicubic': cv2.INTER_CUBIC,
                'bilinear': cv2.INTER_LINEAR,
                'lanczos': cv2.INTER_LANCZOS4,
                'area': cv2.INTER_AREA,
                'nearest': cv2.INTER_NEAREST,
            }
        return self._CV2_INTERP.get(self._interp_name,
                                    self._CV2_INTERP['bicubic'])

    def transform(self, results):
        import cv2

        target_h, target_w = self.scales[np.random.randint(len(self.scales))]
        img = results['img']

        img = cv2.resize(img, (target_w, target_h),
                         interpolation=self._interp())

        pad_h, pad_w = self.max_size
        h, w = img.shape[:2]
        need_h, need_w = pad_h - h, pad_w - w
        if need_h > 0 or need_w > 0:
            img = cv2.copyMakeBorder(
                img, 0, max(need_h, 0), 0, max(need_w, 0),
                cv2.BORDER_CONSTANT, value=self.pad_val)

        results['img'] = img
        results['img_shape'] = (target_h, target_w)
        return results


# ===================================================================
#  Multi-scale Resize + Random Crop (alternative — use when defect
#  region covers most of the image and you want zoom-in augmentation)
# ===================================================================

@TRANSFORMS.register_module()
class MultiScaleResizeCrop(BaseTransform):
    """Resize to a randomly chosen scale, then random-crop to crop_size.

    WARNING: at large upscale ratios, the crop may miss the defect,
    creating label noise.  Prefer ``MultiScaleResize`` for defect
    classification unless defect regions are very large.

    Args:
        scales (list[tuple[int,int]]): Candidate (H, W) resize targets.
        crop_size (tuple[int,int]): Final (H, W) output size.
        interpolation (str): 'bicubic' | 'bilinear' | 'lanczos' | 'area'.
    """

    _INTERP = None

    def __init__(self, scales, crop_size, interpolation='bicubic'):
        super().__init__()
        self.scales = [(h, w) if isinstance(h, int) else (h[0], h[1])
                       for h, w in scales]
        self.crop_size = tuple(crop_size)
        self._interp_name = interpolation

    def _cv2_interp(self):
        if self._INTERP is None:
            import cv2
            MultiScaleResizeCrop._INTERP = {
                'bicubic': cv2.INTER_CUBIC,
                'bilinear': cv2.INTER_LINEAR,
                'lanczos': cv2.INTER_LANCZOS4,
                'area': cv2.INTER_AREA,
                'nearest': cv2.INTER_NEAREST,
            }
        return self._INTERP.get(self._interp_name, self._INTERP['bicubic'])

    def transform(self, results):
        import cv2

        target_h, target_w = self.scales[np.random.randint(len(self.scales))]
        img = results['img']

        img = cv2.resize(img, (target_w, target_h),
                         interpolation=self._cv2_interp())

        crop_h, crop_w = self.crop_size
        h, w = img.shape[:2]
        top = np.random.randint(0, max(h - crop_h, 0) + 1)
        left = np.random.randint(0, max(w - crop_w, 0) + 1)
        img = img[top:top + crop_h, left:left + crop_w]

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        return results


# ===================================================================
#  Metric: Binary Average Precision
# ===================================================================

@METRICS.register_module()
class BinaryAPMetric(BaseMetric):
    """Average Precision using P(defective) as the score.

    Returns ``{'AP': <value in %>}`` compatible with ``CheckpointHook``
    ``save_best='binary_ap/AP'``.
    """

    default_prefix = 'binary_ap'

    @staticmethod
    def _get(sample, key):
        if isinstance(sample, dict):
            return sample[key]
        return getattr(sample, key)

    def process(self, data_batch, data_samples):
        for s in data_samples:
            score = self._get(s, 'pred_score')
            if isinstance(score, torch.Tensor):
                score = score.cpu().numpy()
            label = self._get(s, 'gt_label')
            if isinstance(label, torch.Tensor):
                label = label.item()
            self.results.append(dict(
                pred_score=float(score[1]),
                gt_label=int(label),
            ))

    def compute_metrics(self, results):
        preds = np.array([r['pred_score'] for r in results])
        labels = np.array([r['gt_label'] for r in results])
        if labels.sum() == 0 or labels.sum() == len(labels):
            return dict(AP=0.0)
        ap = average_precision_score(labels, preds)
        return dict(AP=round(ap * 100, 2))


# ===================================================================
#  Metric: Per-Material Average Precision
# ===================================================================

@METRICS.register_module()
class MaterialAPMetric(BaseMetric):
    """AP computed only on samples of a specific ``item_material``.

    Uses ``sample_idx`` from DataSample metainfo to look up each sample's
    material in the parquet file.  Requires the val dataloader to use
    ``shuffle=False`` so that indices align with the parquet row order.

    Args:
        parquet_path (str): Path to the query parquet (same as val dataset).
        material (str): Target material name to evaluate.
    """

    default_prefix = 'material_ap'

    def __init__(self, parquet_path, material='plastic_tight_wrap', **kwargs):
        super().__init__(**kwargs)
        self.material = material
        df = pd.read_parquet(parquet_path)
        self.material_indices = set(
            i for i, (_, row) in enumerate(df.iterrows())
            if row.get('item_material') == material
        )
        total = len(df)
        n_mat = len(self.material_indices)
        print(f'[MaterialAPMetric] Tracking "{material}": '
              f'{n_mat}/{total} samples')

    @staticmethod
    def _get(sample, key):
        if isinstance(sample, dict):
            return sample[key]
        return getattr(sample, key)

    def process(self, data_batch, data_samples):
        for s in data_samples:
            score = self._get(s, 'pred_score')
            if isinstance(score, torch.Tensor):
                score = score.cpu().numpy()
            label = self._get(s, 'gt_label')
            if isinstance(label, torch.Tensor):
                label = label.item()
            idx = self._get(s, 'sample_idx')
            self.results.append(dict(
                pred_score=float(score[1]),
                gt_label=int(label),
                is_target=(idx in self.material_indices),
            ))

    def compute_metrics(self, results):
        target = [r for r in results if r['is_target']]
        if not target:
            return dict(AP=0.0)
        preds = np.array([r['pred_score'] for r in target])
        labels = np.array([r['gt_label'] for r in target])
        if labels.sum() == 0 or labels.sum() == len(labels):
            return dict(AP=0.0)
        ap = average_precision_score(labels, preds)
        return dict(AP=round(ap * 100, 2))
