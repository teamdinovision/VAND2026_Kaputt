"""Kaputt custom modules for MMPretrain — auto-registers on import."""

from .kaputt_dataset import KaputtDataset, MergedKaputtDataset
from .custom_modules import (
    CLSTokenNeck,
    SoftFocalLoss,
    AsymmetricFocalLoss,
    GradualUnfreezeHook,
    GradCheckpointHook,
    SWAHook,
    DefectAwareTransform,
    RandomRotation90,
    BinaryAPMetric,
)

__all__ = [
    'KaputtDataset',
    'MergedKaputtDataset',
    'CLSTokenNeck',
    'SoftFocalLoss',
    'AsymmetricFocalLoss',
    'GradualUnfreezeHook',
    'GradCheckpointHook',
    'SWAHook',
    'DefectAwareTransform',
    'RandomRotation90',
    'BinaryAPMetric',
]
