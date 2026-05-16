"""Kaputt custom modules for MMPretrain — auto-registers on import."""

from .kaputt_dataset import (
    KaputtDataset,
    MergedKaputtDataset,
    KaputtMultiLabelDataset,
    MergedKaputtMultiLabelDataset,
)
from .custom_modules import (
    CLSTokenNeck,
    MultiLayerCLSTokenNeck,
    SoftFocalLoss,
    AsymmetricFocalLoss,
    GradualUnfreezeHook,
    GradCheckpointHook,
    SWAHook,
    DefectAwareTransform,
    RandomRotation90,
    BinaryAPMetric,
    MultiTaskClassifier,
    MultiLayerClassifier,
)

__all__ = [
    'KaputtDataset',
    'MergedKaputtDataset',
    'KaputtMultiLabelDataset',
    'MergedKaputtMultiLabelDataset',
    'CLSTokenNeck',
    'MultiLayerCLSTokenNeck',
    'SoftFocalLoss',
    'AsymmetricFocalLoss',
    'GradualUnfreezeHook',
    'GradCheckpointHook',
    'SWAHook',
    'DefectAwareTransform',
    'RandomRotation90',
    'BinaryAPMetric',
    'MultiTaskClassifier',
    'MultiLayerClassifier',
]
