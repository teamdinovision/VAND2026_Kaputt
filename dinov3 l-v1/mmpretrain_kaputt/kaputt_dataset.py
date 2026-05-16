"""Kaputt binary defect dataset — reads Parquet annotations for MMPretrain.

Provides:
  - KaputtDataset                 : Standard single-split dataset
  - MergedKaputtDataset           : Merge multiple splits (train+val) with optional
                                    repeat-factor oversampling for rare material classes
  - ReferenceEnhancedKaputtDataset: Query data + confidence-filtered pseudo-labeled
                                    reference data for noise-robust training
  - ExternalEnhancedKaputtDataset : Query + pseudo-labeled reference + external datasets
                                    (e.g. ARMBench) for broader domain coverage

When auxiliary tasks are enabled, gt_label is an 11-dim float vector:
  [binary,
   penetration, deformation, actuation, deconstruction,
   spillage, superficial, missing_unit,
   has_defect_type_flag,
   material_id, has_material_flag]

The MultiAuxHead only computes each auxiliary loss where its flag == 1.
"""

import os
import numpy as np
import pandas as pd
from mmengine.dataset import BaseDataset
from mmpretrain.registry import DATASETS

# ── Defect types (multi-label, 7 classes) ──────────────────────────
NUM_DEFECT_TYPES = 7
DEFECT_TYPE_INDICES = {
    'penetration': 0,
    'deformation': 1,
    'actuation': 2,
    'deconstruction': 3,
    'spillage': 4,
    'superficial': 5,
    'missing_unit': 6,
}

ARMBENCH_TYPE_MAP = {
    'open': 'actuation',
    'deconstruction': 'deconstruction',
}

# ── Material types (single-label, 10 classes) ─────────────────────
NUM_MATERIALS = 10
MATERIAL_INDICES = {
    'cardboard': 0,
    'plastic_loose_bag': 1,
    'plastic_hard': 2,
    'plastic_tight_wrap': 3,
    'plastic_bubble_wrap': 4,
    'book_paper': 5,
    'book_other': 6,
    'book_plastic_tight_wrap': 7,
    'paper': 8,
    'other': 9,
}

# ── Compound label layout (11-dim) ────────────────────────────────
# [0]     binary
# [1:8]   defect_types multi-hot  (7 dim)
# [8]     has_defect_type flag
# [9]     material_id             (int encoded as float, 0-9)
# [10]    has_material flag
LABEL_DIM = 11


def _parse_defect_types(defect_types_str):
    """Convert comma-separated defect type string to 7-dim multi-hot vector."""
    vec = np.zeros(NUM_DEFECT_TYPES, dtype=np.float32)
    if not defect_types_str or str(defect_types_str) == 'nan':
        return vec
    for dt in str(defect_types_str).split(','):
        dt = dt.strip()
        if dt in DEFECT_TYPE_INDICES:
            vec[DEFECT_TYPE_INDICES[dt]] = 1.0
    return vec


def _make_label(binary, defect_types_vec=None, has_dt=False,
                material_id=-1, has_mat=False):
    """Build 11-dim compound label."""
    label = np.zeros(LABEL_DIM, dtype=np.float32)
    label[0] = float(binary)
    if defect_types_vec is not None:
        label[1:1 + NUM_DEFECT_TYPES] = defect_types_vec
    if has_dt:
        label[8] = 1.0
    if has_mat and material_id >= 0:
        label[9] = float(material_id)
        label[10] = 1.0
    return label


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
#  Query + Pseudo-labeled Reference — noise-robust training
# ===================================================================

@DATASETS.register_module()
class ReferenceEnhancedKaputtDataset(BaseDataset):
    """Query data + confidence-filtered pseudo-labeled reference data.

    Combines ground-truth query labels with teacher-model pseudo-labels for
    reference images.  Only reference samples where the teacher is
    sufficiently confident are included, preventing noisy labels from
    degrading performance.

    The pseudo-label CSV is generated by ``generate_pseudo_labels.py`` and
    contains per-image defect probabilities from the teacher model.

    Args:
        ann_files (list[str]): Query parquet paths (one per split).
        data_prefixes (list[str]): Image root dirs for query data.
        pseudo_label_file (str): CSV from generate_pseudo_labels.py.
        conf_thresh_normal (float): Include reference as normal if
            defect_prob < this value.  Default 0.2.
        conf_thresh_defect (float): Include reference as defective if
            defect_prob > this value.  Default 0.8.
        max_ref_ratio (float): Cap reference count at this multiple of
            query count.  Prevents reference data from overwhelming the
            batch.  -1 = unlimited.  Default 1.0.
        ref_repeat_defect (int): Repeat each pseudo-defective reference
            sample this many times.  Helps balance the large number of
            pseudo-normal samples.  Default 1.
        pipeline (list[dict]): Transform pipeline.
    """

    METAINFO = {'classes': ('non_defective', 'defective')}

    def __init__(self, ann_files, data_prefixes,
                 pseudo_label_file='',
                 conf_thresh_normal=0.2,
                 conf_thresh_defect=0.8,
                 max_ref_ratio=1.0,
                 ref_repeat_defect=1,
                 pipeline=(), **kwargs):
        self._ann_files = ann_files
        self._data_prefixes = data_prefixes
        self._pseudo_label_file = pseudo_label_file
        self._conf_normal = conf_thresh_normal
        self._conf_defect = conf_thresh_defect
        self._max_ref_ratio = max_ref_ratio
        self._ref_repeat_defect = ref_repeat_defect
        super().__init__(
            ann_file=ann_files[0], pipeline=list(pipeline), **kwargs)

    def load_data_list(self):
        # --- query data (ground-truth labels) ---
        query_list = []
        for ann, prefix in zip(self._ann_files, self._data_prefixes):
            df = pd.read_parquet(ann)
            for _, row in df.iterrows():
                query_list.append(dict(
                    img_path=os.path.join(prefix, f'{row.capture_id}.jpg'),
                    gt_label=int(row.defect),
                ))
        n_query = len(query_list)

        # --- reference data (pseudo-labels from teacher model) ---
        ref_normal, ref_defect, n_skip = [], [], 0
        if self._pseudo_label_file and os.path.exists(self._pseudo_label_file):
            ref_df = pd.read_csv(self._pseudo_label_file)
            for _, row in ref_df.iterrows():
                p = row['defect_prob']
                if p < self._conf_normal:
                    ref_normal.append(dict(
                        img_path=row['img_path'], gt_label=0))
                elif p > self._conf_defect:
                    ref_defect.append(dict(
                        img_path=row['img_path'], gt_label=1))
                else:
                    n_skip += 1
        else:
            if self._pseudo_label_file:
                print(f'[RefEnhancedDataset] WARNING: pseudo_label_file '
                      f'not found: {self._pseudo_label_file}')

        ref_list = ref_normal.copy()
        for _ in range(max(1, self._ref_repeat_defect)):
            ref_list.extend(ref_defect)

        if self._max_ref_ratio > 0 and ref_list:
            cap = int(n_query * self._max_ref_ratio)
            if len(ref_list) > cap:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(ref_list), cap, replace=False)
                ref_list = [ref_list[i] for i in sorted(idx)]

        data_list = query_list + ref_list
        n_ref = len(ref_list)
        n_ref_norm = sum(1 for r in ref_list if r['gt_label'] == 0)
        n_ref_def = n_ref - n_ref_norm

        print(f'[RefEnhancedDataset] Query={n_query}  '
              f'Ref={n_ref} (normal={n_ref_norm}, defect={n_ref_def}, '
              f'skipped={n_skip})  Total={len(data_list)}')
        return data_list


# ===================================================================
#  Query + Pseudo-labeled Reference + External data (e.g. ARMBench)
# ===================================================================

@DATASETS.register_module()
class ExternalEnhancedKaputtDataset(BaseDataset):
    """Kaputt query data + pseudo-labeled reference + external datasets.

    Extends ReferenceEnhancedKaputtDataset by adding support for external
    defect-detection datasets (ARMBench, MVTec-AD, etc.) provided as CSV
    files with ``img_path`` and ``gt_label`` columns.

    When ``enable_defect_types`` or ``enable_material`` is True, gt_label
    becomes an 11-dim float vector encoding binary + defect types + material.
    Only samples with the relevant annotations have their flags set to 1;
    other sources get flag=0 so the auxiliary heads skip them.

    Args:
        ann_files (list[str]): Kaputt query parquet paths (one per split).
        data_prefixes (list[str]): Image root dirs for query data.
        pseudo_label_file (str): CSV from generate_pseudo_labels.py.
        external_csv_files (list[str]): CSVs from prepare_armbench.py or
            similar.  Must have ``img_path`` and ``gt_label`` columns.
        conf_thresh_normal (float): Include reference as normal if
            defect_prob < this value.
        conf_thresh_defect (float): Include reference as defective if
            defect_prob > this value.
        max_ref_ratio (float): Cap reference count at this ratio of
            query count.  -1 = unlimited.
        ref_repeat_defect (int): Repeat pseudo-defective references.
        external_ratio (float): Cap external data at this ratio of
            query count.  -1 = unlimited.  Default 0.5.
        external_defect_repeat (int): Repeat external defective samples
            to compensate for class imbalance.  Default 1.
        enable_defect_types (bool): If True, encode defect-type multi-label
            in gt_label for the auxiliary defect-type head.  Default False.
        enable_material (bool): If True, encode item_material class id
            in gt_label for the auxiliary material head.  Default False.
        map_external_types (bool): If True AND enable_defect_types, map
            ARMBench defect_type column to Kaputt types using
            ARMBENCH_TYPE_MAP.  Default False.
        pipeline (list[dict]): Transform pipeline.
    """

    METAINFO = {'classes': ('non_defective', 'defective')}

    def __init__(self, ann_files, data_prefixes,
                 pseudo_label_file='',
                 external_csv_files=None,
                 conf_thresh_normal=0.2,
                 conf_thresh_defect=0.8,
                 max_ref_ratio=1.0,
                 ref_repeat_defect=1,
                 external_ratio=0.5,
                 external_defect_repeat=1,
                 enable_defect_types=False,
                 enable_material=False,
                 map_external_types=False,
                 pipeline=(), **kwargs):
        self._ann_files = ann_files
        self._data_prefixes = data_prefixes
        self._pseudo_label_file = pseudo_label_file
        self._external_csv_files = external_csv_files or []
        self._conf_normal = conf_thresh_normal
        self._conf_defect = conf_thresh_defect
        self._max_ref_ratio = max_ref_ratio
        self._ref_repeat_defect = ref_repeat_defect
        self._external_ratio = external_ratio
        self._external_defect_repeat = external_defect_repeat
        self._enable_dt = enable_defect_types
        self._enable_mat = enable_material
        self._map_ext_types = map_external_types
        super().__init__(
            ann_file=ann_files[0], pipeline=list(pipeline), **kwargs)

    @property
    def _use_compound(self):
        return self._enable_dt or self._enable_mat

    def _make_gt(self, binary, dt_vec=None, has_dt=False,
                 material_id=-1, has_mat=False):
        if not self._use_compound:
            return int(binary)
        return _make_label(binary, dt_vec, has_dt, material_id, has_mat)

    def load_data_list(self):
        # --- 1. Kaputt query data (ground-truth labels) ---
        query_list = []
        n_with_dt, n_with_mat = 0, 0
        for ann, prefix in zip(self._ann_files, self._data_prefixes):
            df = pd.read_parquet(ann)
            has_dt_col = 'defect_types' in df.columns
            has_mat_col = 'item_material' in df.columns
            for _, row in df.iterrows():
                binary = int(row.defect)

                dt_vec, has_dt = None, False
                if self._enable_dt and has_dt_col and binary == 1:
                    dt_vec = _parse_defect_types(
                        getattr(row, 'defect_types', ''))
                    has_dt = dt_vec.sum() > 0
                    if has_dt:
                        n_with_dt += 1
                elif self._enable_dt and binary == 0:
                    dt_vec = np.zeros(NUM_DEFECT_TYPES, dtype=np.float32)
                    has_dt = True
                    n_with_dt += 1

                material_id, has_mat = -1, False
                if self._enable_mat and has_mat_col:
                    mat_str = str(getattr(row, 'item_material', ''))
                    if mat_str in MATERIAL_INDICES:
                        material_id = MATERIAL_INDICES[mat_str]
                        has_mat = True
                        n_with_mat += 1

                query_list.append(dict(
                    img_path=os.path.join(prefix, f'{row.capture_id}.jpg'),
                    gt_label=self._make_gt(
                        binary, dt_vec, has_dt, material_id, has_mat),
                ))
        n_query = len(query_list)

        # --- 2. Kaputt reference data (pseudo-labels, no aux labels) ---
        ref_normal, ref_defect, n_skip = [], [], 0
        if self._pseudo_label_file and os.path.exists(self._pseudo_label_file):
            ref_df = pd.read_csv(self._pseudo_label_file)
            for _, row in ref_df.iterrows():
                p = row['defect_prob']
                if p < self._conf_normal:
                    ref_normal.append(dict(
                        img_path=row['img_path'],
                        gt_label=self._make_gt(0)))
                elif p > self._conf_defect:
                    ref_defect.append(dict(
                        img_path=row['img_path'],
                        gt_label=self._make_gt(1)))
                else:
                    n_skip += 1
        elif self._pseudo_label_file:
            print(f'[ExternalEnhanced] WARNING: pseudo_label_file '
                  f'not found: {self._pseudo_label_file}')

        ref_list = ref_normal.copy()
        for _ in range(max(1, self._ref_repeat_defect)):
            ref_list.extend(ref_defect)

        if self._max_ref_ratio > 0 and ref_list:
            cap = int(n_query * self._max_ref_ratio)
            if len(ref_list) > cap:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(ref_list), cap, replace=False)
                ref_list = [ref_list[i] for i in sorted(idx)]

        n_ref = len(ref_list)
        n_ref_norm = sum(1 for r in ref_list
                         if (r['gt_label'] if np.isscalar(r['gt_label'])
                             else r['gt_label'][0]) == 0)
        n_ref_def = n_ref - n_ref_norm

        # --- 3. External datasets (e.g. ARMBench) ---
        ext_list = []
        n_ext_with_dt = 0
        for csv_path in self._external_csv_files:
            if not os.path.exists(csv_path):
                print(f'[ExternalEnhanced] WARNING: external CSV '
                      f'not found: {csv_path}')
                continue
            ext_df = pd.read_csv(csv_path)
            has_type_col = 'defect_type' in ext_df.columns
            ext_normal, ext_defect = [], []
            for _, row in ext_df.iterrows():
                binary = int(row['gt_label'])
                dt_vec, has_dt = None, False

                if (self._enable_dt and self._map_ext_types
                        and has_type_col and binary == 1):
                    raw_type = str(row.get('defect_type', ''))
                    mapped = ARMBENCH_TYPE_MAP.get(raw_type, '')
                    if mapped and mapped in DEFECT_TYPE_INDICES:
                        dt_vec = np.zeros(
                            NUM_DEFECT_TYPES, dtype=np.float32)
                        dt_vec[DEFECT_TYPE_INDICES[mapped]] = 1.0
                        has_dt = True
                        n_ext_with_dt += 1

                item = dict(
                    img_path=str(row['img_path']),
                    gt_label=self._make_gt(binary, dt_vec, has_dt),
                )
                if binary == 0:
                    ext_normal.append(item)
                else:
                    ext_defect.append(item)

            src_list = ext_normal.copy()
            for _ in range(max(1, self._external_defect_repeat)):
                src_list.extend(ext_defect)

            print(f'[ExternalEnhanced] {csv_path}: '
                  f'normal={len(ext_normal)}, defect={len(ext_defect)}, '
                  f'after repeat={len(src_list)}')
            ext_list.extend(src_list)

        if self._external_ratio > 0 and ext_list:
            cap = int(n_query * self._external_ratio)
            if len(ext_list) > cap:
                rng = np.random.RandomState(123)
                idx = rng.choice(len(ext_list), cap, replace=False)
                ext_list = [ext_list[i] for i in sorted(idx)]

        n_ext = len(ext_list)
        n_ext_norm = sum(1 for e in ext_list
                         if (e['gt_label'] if np.isscalar(e['gt_label'])
                             else e['gt_label'][0]) == 0)
        n_ext_def = n_ext - n_ext_norm

        # --- Combine all sources ---
        data_list = query_list + ref_list + ext_list

        aux_info = ''
        if self._enable_dt:
            aux_info += (f'  DefectTypes: query={n_with_dt}, '
                         f'ext_mapped={n_ext_with_dt}')
        if self._enable_mat:
            aux_info += f'  Material: query={n_with_mat}'

        print(f'[ExternalEnhanced] '
              f'Query={n_query}  '
              f'Ref={n_ref} (norm={n_ref_norm}, def={n_ref_def}, '
              f'skip={n_skip})  '
              f'External={n_ext} (norm={n_ext_norm}, def={n_ext_def})  '
              f'Total={len(data_list)}{aux_info}')
        return data_list
