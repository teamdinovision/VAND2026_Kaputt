"""Kaputt binary defect dataset — reads Parquet annotations for MMPretrain.

Provides:
  - KaputtDataset              : Standard single-split dataset
  - MergedKaputtDataset        : Merge multiple splits (train+val) with optional
                                 repeat-factor oversampling for rare material classes
  - KaputtMultiLabelDataset    : Single-split with defect-type multi-label
  - MergedKaputtMultiLabelDataset : Merged splits with defect-type multi-label
"""

import os
import pandas as pd
from mmengine.dataset import BaseDataset
from mmpretrain.registry import DATASETS

DEFECT_TYPES = [
    'deformation', 'actuation', 'deconstruction',
    'penetration', 'superficial', 'spillage', 'missing_unit',
]
NUM_DEFECT_TYPES = len(DEFECT_TYPES)
_DTYPE_TO_IDX = {t: i for i, t in enumerate(DEFECT_TYPES)}


def _parse_defect_types(defect_types_str):
    """Convert comma-separated defect_types string to multi-hot list."""
    vec = [0] * NUM_DEFECT_TYPES
    if not defect_types_str:
        return vec
    for t in defect_types_str.split(','):
        t = t.strip()
        if t in _DTYPE_TO_IDX:
            vec[_DTYPE_TO_IDX[t]] = 1
    return vec


@DATASETS.register_module()
class KaputtDataset(BaseDataset):
    """Binary classification dataset (defective vs non-defective).

    Expects a Parquet file with at least ``capture_id`` and ``defect`` columns.
    Images are looked up as ``<data_prefix.img_path>/<capture_id>.jpg``.
    """

    METAINFO = {'classes': ('non_defective', 'defective')}

    def load_data_list(self):
        df = pd.read_parquet(self.ann_file)
        img_prefix = self.data_prefix.get('img_path', '')
        data_list = []
        for _, row in df.iterrows():
            data_list.append(dict(
                img_path=os.path.join(img_prefix, f'{row.capture_id}.jpg'),
                gt_label=int(row.defect),
            ))
        return data_list


# ===================================================================
#  [ABLATION] Trick: Merged Train+Val with Repeat-Factor Oversampling
#  To disable repeat-factor: set repeat_factors={} in config
#  To disable merge: use KaputtDataset with single ann_file instead
# ===================================================================

@DATASETS.register_module()
class MergedKaputtDataset(BaseDataset):
    """Merge multiple dataset splits with repeat-factor oversampling.

    Concatenates data from multiple Parquet files (e.g. train + validation)
    and optionally duplicates samples from rare ``item_material`` classes
    to address class imbalance.

    Args:
        ann_files (list[str]): Parquet file paths for each split.
        data_prefixes (list[str]): Image root directories, one per split.
        repeat_factors (dict[str, int]): Material-type → repeat count.
            E.g. ``{'book_other': 5, 'other': 4, 'plastic_tight_wrap': 3}``.
            Materials not listed default to repeat factor 1.
        pipeline (list[dict]): Transform pipeline.
    """

    METAINFO = {'classes': ('non_defective', 'defective')}

    def __init__(self, ann_files, data_prefixes, repeat_factors=None,
                 pipeline=(), **kwargs):
        self._multi_ann_files = ann_files
        self._multi_data_prefixes = data_prefixes
        self._repeat_factors = repeat_factors or {}
        super().__init__(
            ann_file=ann_files[0], pipeline=list(pipeline), **kwargs)

    def load_data_list(self):
        data_list = []
        total_original = 0
        total_repeated = 0

        for ann_file, prefix in zip(
                self._multi_ann_files, self._multi_data_prefixes):
            df = pd.read_parquet(ann_file)
            for _, row in df.iterrows():
                material = str(getattr(row, 'item_material', ''))
                item = dict(
                    img_path=os.path.join(prefix, f'{row.capture_id}.jpg'),
                    gt_label=int(row.defect),
                )
                repeat = self._repeat_factors.get(material, 1)
                total_original += 1
                total_repeated += repeat
                for _ in range(repeat):
                    data_list.append(item.copy())

        print(f'[MergedKaputtDataset] Loaded {total_original} samples '
              f'from {len(self._multi_ann_files)} splits, '
              f'expanded to {total_repeated} with repeat factors: '
              f'{self._repeat_factors}')
        return data_list


# ===================================================================
#  Multi-label variants: include defect_type_label for auxiliary head
# ===================================================================

@DATASETS.register_module()
class KaputtMultiLabelDataset(BaseDataset):
    """Binary classification dataset with defect-type multi-label.

    Each sample carries both the binary ``gt_label`` (0/1) and a
    ``defect_type_label`` multi-hot vector of 7 defect types.
    Use ``PackInputs(algorithm_keys=('defect_type_label',))`` to
    pass the multi-label through to DataSample.
    """

    METAINFO = {'classes': ('non_defective', 'defective')}

    def load_data_list(self):
        df = pd.read_parquet(self.ann_file)
        img_prefix = self.data_prefix.get('img_path', '')
        data_list = []
        for _, row in df.iterrows():
            dt_str = str(getattr(row, 'defect_types', '') or '')
            data_list.append(dict(
                img_path=os.path.join(img_prefix, f'{row.capture_id}.jpg'),
                gt_label=int(row.defect),
                defect_type_label=_parse_defect_types(dt_str),
            ))
        return data_list


@DATASETS.register_module()
class MergedKaputtMultiLabelDataset(BaseDataset):
    """Merged multi-split dataset with defect-type multi-label.

    Same as MergedKaputtDataset but additionally outputs
    ``defect_type_label`` for auxiliary multi-label supervision.
    """

    METAINFO = {'classes': ('non_defective', 'defective')}

    def __init__(self, ann_files, data_prefixes, repeat_factors=None,
                 pipeline=(), **kwargs):
        self._multi_ann_files = ann_files
        self._multi_data_prefixes = data_prefixes
        self._repeat_factors = repeat_factors or {}
        super().__init__(
            ann_file=ann_files[0], pipeline=list(pipeline), **kwargs)

    def load_data_list(self):
        data_list = []
        total_original = 0
        total_repeated = 0

        for ann_file, prefix in zip(
                self._multi_ann_files, self._multi_data_prefixes):
            df = pd.read_parquet(ann_file)
            for _, row in df.iterrows():
                material = str(getattr(row, 'item_material', ''))
                dt_str = str(getattr(row, 'defect_types', '') or '')
                item = dict(
                    img_path=os.path.join(prefix, f'{row.capture_id}.jpg'),
                    gt_label=int(row.defect),
                    defect_type_label=_parse_defect_types(dt_str),
                )
                repeat = self._repeat_factors.get(material, 1)
                total_original += 1
                total_repeated += repeat
                for _ in range(repeat):
                    data_list.append(item.copy())

        print(f'[MergedKaputtMultiLabelDataset] Loaded {total_original} '
              f'samples from {len(self._multi_ann_files)} splits, '
              f'expanded to {total_repeated} with repeat factors: '
              f'{self._repeat_factors}')
        return data_list
