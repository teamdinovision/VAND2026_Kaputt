"""Kaputt custom modules for MMPretrain — auto-registers on import."""

from .kaputt_dataset import (
    KaputtDataset,
    MergedKaputtDataset,
    ReferenceEnhancedKaputtDataset,
    ExternalEnhancedKaputtDataset,
)
from .custom_modules import (
    CLSTokenNeck,
    SoftFocalLoss,
    AsymmetricFocalLoss,
    NoisyRobustFocalLoss,
    AuxDefectTypeHead,
    MultiAuxHead,
    GradualUnfreezeHook,
    GradCheckpointHook,
    SWAHook,
    DefectAwareTransform,
    RandomRotation90,
    BinaryAPMetric,
    MaterialAPMetric,
)

__all__ = [
    'KaputtDataset',
    'MergedKaputtDataset',
    'ReferenceEnhancedKaputtDataset',
    'ExternalEnhancedKaputtDataset',
    'CLSTokenNeck',
    'SoftFocalLoss',
    'AsymmetricFocalLoss',
    'NoisyRobustFocalLoss',
    'AuxDefectTypeHead',
    'MultiAuxHead',
    'GradualUnfreezeHook',
    'GradCheckpointHook',
    'SWAHook',
    'DefectAwareTransform',
    'RandomRotation90',
    'BinaryAPMetric',
    'MaterialAPMetric',
]
