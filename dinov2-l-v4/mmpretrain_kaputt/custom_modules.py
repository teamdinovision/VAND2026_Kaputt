"""
Custom modules for Kaputt defect classification with MMPretrain.

Includes:
  - CLSTokenNeck           : CLS token pooling for TIMMBackbone ViT output
  - MultiLayerCLSTokenNeck : CLS token pooling for multi-layer ViT outputs
  - SoftFocalLoss          : Focal Loss + OHEM, supports soft labels (Mixup/CutMix)
  - AsymmetricLoss         : Asymmetric focal loss for AP-optimal binary classification
  - GradualUnfreezeHook    : Progressive backbone layer unfreezing (optional)
  - SWAHook                : Stochastic Weight Averaging
  - DefectAwareTransform   : Stronger augmentation for defective samples
  - RandomRotation90       : Safe 90-degree rotation augmentation
  - BinaryAPMetric         : Average Precision metric for binary classification
  - MaterialAPMetric       : AP for a specific item_material subset
  - MultiTaskClassifier    : Binary + auxiliary multi-label defect-type head
  - MultiLayerClassifier   : Multi-task with intermediate layer supervision (deep supervision)
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
from mmpretrain.models.classifiers.image import ImageClassifier
from mmpretrain.models.heads.cls_head import ClsHead


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


@MODELS.register_module()
class MultiLayerCLSTokenNeck(nn.Module):
    """Extract CLS token or pool features from multiple ViT layers.
    
    Compatible with multiple output formats:
      - 3D token sequence: (B, N, D) -> (B, D) via CLS token (index 0)
      - 4D spatial feature map: (B, C, H, W) -> (B, C) via global avg pool
        (produced by TIMMBackbone with features_only=True)
      - 2D already pooled: (B, D) -> (B, D) passthrough
    
    This neck is designed for deep supervision where classification heads
    are applied to intermediate transformer layers.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _pool_single(x):
        if x.dim() == 3:
            return x[:, 0]  # CLS token
        elif x.dim() == 4:
            return x.mean(dim=(2, 3))  # Global average pooling
        return x  # Already (B, D)

    def forward(self, inputs):
        if isinstance(inputs, (tuple, list)):
            return tuple(self._pool_single(x) for x in inputs)
        return (self._pool_single(inputs),)


# ===================================================================
#  Neck: CLS + Patch Average Pooling (global + local information)
# ===================================================================

@MODELS.register_module()
class CLSPatchNeck(nn.Module):
    """Concatenate CLS token and global-average-pooled patch tokens.

    Produces (B, 2D) by combining global semantics (CLS) with local
    spatial information (patch avg). Particularly effective for defect
    detection where defects are local anomalies.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        if isinstance(inputs, (tuple, list)):
            x = inputs[-1]
        else:
            x = inputs
        if x.dim() == 3:
            cls_token = x[:, 0]
            patch_avg = x[:, 1:].mean(dim=1)
            return (torch.cat([cls_token, patch_avg], dim=1),)
        return (x,)


# ===================================================================
#  Head: MLP Classification Head (nonlinear, better than LinearClsHead)
# ===================================================================

@MODELS.register_module()
class MLPClsHead(ClsHead):
    """MLP classification head: LayerNorm -> Linear -> GELU -> Dropout -> Linear.

    Provides better nonlinear feature mapping than LinearClsHead,
    especially for DINOv2 where the CLS token feature space is highly
    nonlinear. Typically yields +0.3-0.8% over a single linear layer.

    Args:
        num_classes (int): Number of output classes.
        in_channels (int): Input feature dimension.
        hidden_channels (int): Hidden layer dimension. Defaults to 64.
        dropout_rate (float): Dropout probability in the hidden layer.
    """

    def __init__(self, num_classes, in_channels, hidden_channels=64,
                 dropout_rate=0.1, init_cfg=None, **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            # nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, num_classes),
        )

    def pre_logits(self, feats):
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        return feats

    def forward(self, feats):
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        return self.mlp(feats)


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


# ===================================================================
#  Multi-task classifier: binary head + defect-type auxiliary head
#  Inference uses ONLY the binary head (no overhead at test time).
# ===================================================================

@MODELS.register_module()
class MultiTaskClassifier(ImageClassifier):
    """ImageClassifier with an auxiliary multi-label defect-type head.

    Shares backbone + neck between the primary binary head and a lightweight
    auxiliary linear layer that predicts 7 defect-type categories via BCE.
    The auxiliary loss acts as a regularizer during training and is
    discarded at inference time.

    Fully inherits ImageClassifier so train_step / val_step / predict /
    forward(mode='tensor') all work unchanged.  Only ``loss()`` is
    overridden to add the auxiliary BCE term.

    Args:
        aux_head (dict): Must contain ``in_channels`` and ``num_classes``.
            Optional ``pos_weight`` (list[float]) for BCE class balancing.
        aux_loss_weight (float): Scalar multiplier for the auxiliary loss.
        **kwargs: Forwarded to ImageClassifier (backbone, neck, head, ...).
    """

    def __init__(self, aux_head, aux_loss_weight=0.3, **kwargs):
        super().__init__(**kwargs)
        self.aux_fc = nn.Linear(aux_head['in_channels'],
                                aux_head['num_classes'])
        self.aux_loss_weight = aux_loss_weight
        pos_weight = aux_head.get('pos_weight', None)
        if pos_weight is not None:
            self.register_buffer(
                'aux_pos_weight',
                torch.tensor(pos_weight, dtype=torch.float))
        else:
            self.aux_pos_weight = None

    def loss(self, inputs, data_samples):
        feats = self.extract_feat(inputs)
        losses = self.head.loss(feats, data_samples)

        cls_feat = feats[-1] if isinstance(feats, (tuple, list)) else feats
        aux_logits = self.aux_fc(cls_feat)

        aux_targets = []
        for ds in data_samples:
            dt = ds.get('defect_type_label')
            if dt is None:
                dt = [0] * aux_logits.size(1)
            if isinstance(dt, torch.Tensor):
                aux_targets.append(dt.float())
            else:
                aux_targets.append(torch.tensor(dt, dtype=torch.float32))
        aux_targets = torch.stack(aux_targets).to(aux_logits.device)

        aux_loss = F.binary_cross_entropy_with_logits(
            aux_logits, aux_targets, pos_weight=self.aux_pos_weight)
        losses['aux_loss'] = aux_loss * self.aux_loss_weight
        return losses


@MODELS.register_module()
class MultiLayerClassifier(MultiTaskClassifier):
    """MultiTaskClassifier with intermediate layer supervision (deep supervision).
    
    Extends MultiTaskClassifier to support classification heads at multiple
    intermediate transformer layers, enabling better gradient flow and
    multi-scale feature learning.
    
    Features:
      - Fully backward-compatible: if intermediate_heads=None, behaves
        exactly like MultiTaskClassifier
      - Supports multiple intermediate layer heads with configurable loss weights
      - Shares backbone features across all heads for efficiency
      - Uses deep supervision to improve training convergence
    
    Args:
        intermediate_heads (dict, optional): Dict mapping layer names to head configs.
            Example: {'layer_15': {...}, 'layer_19': {...}}
            Each head config should be a standard LinearClsHead config dict.
            If None, behaves identically to MultiTaskClassifier.
        **kwargs: All other arguments passed to MultiTaskClassifier.
    
    Example config:
        model = dict(
            type='MultiLayerClassifier',
            backbone=dict(
                type='TIMMBackbone',
                model_name='vit_large_patch14_dinov2.lvd142m',
                features_only=True,
                out_indices=(15, 19, 23),  # layer indices to extract
            ),
            neck=dict(type='MultiLayerCLSTokenNeck'),
            head=dict(
                type='LinearClsHead',
                num_classes=2,
                in_channels=1024,
                loss=dict(type='SoftFocalLoss', loss_weight=1.0),
            ),
            aux_head=dict(num_classes=7, in_channels=1024, pos_weight=[...]),
            intermediate_heads=dict(
                layer_15=dict(
                    type='LinearClsHead',
                    num_classes=2,
                    in_channels=1024,
                    loss=dict(type='SoftFocalLoss', loss_weight=0.3),
                ),
                layer_19=dict(
                    type='LinearClsHead',
                    num_classes=2,
                    in_channels=1024,
                    loss=dict(type='SoftFocalLoss', loss_weight=0.5),
                ),
            ),
        )
    """
    
    def __init__(self, intermediate_heads=None, **kwargs):
        super().__init__(**kwargs)
        
        # Build intermediate layer heads
        self.intermediate_heads = nn.ModuleDict()
        if intermediate_heads:
            for layer_name, head_cfg in intermediate_heads.items():
                self.intermediate_heads[layer_name] = MODELS.build(head_cfg)
    
    def loss(self, inputs, data_samples):
        """Compute loss with intermediate layer supervision.
        
        If intermediate_heads is None or empty, falls back to parent behavior.
        Otherwise, computes losses for all intermediate heads plus the main
        and auxiliary heads.
        
        Memory optimization: Intermediate layer losses are computed sequentially
        to avoid holding all features in memory simultaneously.
        """
        # Extract features from all layers
        feats = self.extract_feat(inputs)
        
        # Handle single-layer output (backward compatibility)
        if not isinstance(feats, (tuple, list)):
            feats = (feats,)
        
        # Initialize loss dict
        losses = {}
        
        # Intermediate layer heads (if configured)
        # Compute these FIRST to allow earlier feature maps to be freed
        if self.intermediate_heads and len(feats) > 1:
            for i, (layer_name, head) in enumerate(self.intermediate_heads.items()):
                # Use the i-th feature for the i-th intermediate head
                # (assumes features are ordered from shallow to deep)
                if i < len(feats) - 1:  # exclude final layer (already used for main head)
                    inter_feat = (feats[i],)
                    inter_losses = head.loss(inter_feat, data_samples)
                    # Prefix loss keys with layer name to distinguish them
                    for k, v in inter_losses.items():
                        losses[f'{layer_name}_{k}'] = v
        
        # Main classification head (uses final layer feature)
        final_feat = (feats[-1],)
        main_losses = self.head.loss(final_feat, data_samples)
        losses.update(main_losses)
        
        # Auxiliary defect-type head (uses final layer feature)
        cls_feat = feats[-1]
        aux_logits = self.aux_fc(cls_feat)
        
        aux_targets = []
        for ds in data_samples:
            dt = ds.get('defect_type_label')
            if dt is None:
                dt = [0] * aux_logits.size(1)
            if isinstance(dt, torch.Tensor):
                aux_targets.append(dt.float())
            else:
                aux_targets.append(torch.tensor(dt, dtype=torch.float32))
        aux_targets = torch.stack(aux_targets).to(aux_logits.device)
        
        aux_loss = F.binary_cross_entropy_with_logits(
            aux_logits, aux_targets, pos_weight=self.aux_pos_weight)
        losses['aux_loss'] = aux_loss * self.aux_loss_weight
        
        return losses
