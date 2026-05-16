"""Kaputt DINOv2 — MLP Head experiment.

Based on large_728.py with LinearClsHead replaced by MLPClsHead:
  LayerNorm -> Linear(D, D) -> GELU -> Dropout(0.1) -> Linear(D, 2)

DINOv2 CLS token features are highly nonlinear; a single linear layer
has limited expressiveness.  The MLP head adds a hidden layer with
normalization and nonlinearity for better feature utilization.
"""

# ======================== Backbone Selection ========================
backbone_name = 'dinov2_l'

_REGISTRY = dict(
    dinov2_b=dict(timm='vit_base_patch14_dinov2.lvd142m',           ch=768,  ps=14),
    dinov2_l=dict(timm='vit_large_patch14_dinov2.lvd142m',          ch=1024, ps=14),
    dinov2_l_reg=dict(timm='vit_large_patch14_reg4_dinov2.lvd142m', ch=1024, ps=14),
    dinov2_g=dict(timm='vit_giant_patch14_dinov2.lvd142m',          ch=1536, ps=14),
)
_bb = _REGISTRY[backbone_name]

# ======================== Image Size ================================
target_img_size = 728
patch_size = _bb['ps']
img_size = (target_img_size // patch_size) * patch_size

# ======================== Data Paths ================================
data_root = "/data/public/dataset/kaputt"
parquet_root = "/data/public/dataset/kaputt"

# ======================== Runtime ===================================
default_scope = 'mmpretrain'
work_dir = f'./work_dirs/large_728_mlp'

# ======================== Model =====================================
model = dict(
    type='MultiTaskClassifier',
    data_preprocessor=dict(
        num_classes=2,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    backbone=dict(
        type='TIMMBackbone',
        model_name=_bb['timm'],
        pretrained=True,
        img_size=img_size,
    ),
    neck=dict(type='CLSTokenNeck'),
    head=dict(
        type='MLPClsHead',
        num_classes=2,
        in_channels=_bb['ch'],
        hidden_channels=64,
        dropout_rate=0.0,
        loss=dict(
            type='SoftFocalLoss',
            gamma=2.0,
            alpha=0.75,
            ohem_ratio=0.7,
            loss_weight=1.0,
        ),
    ),
    aux_head=dict(
        num_classes=7,
        in_channels=_bb['ch'],
    ),
    aux_loss_weight=0.3,
)

model_wrapper_cfg = dict(find_unused_parameters=True)

# ======================== Data Pipelines ============================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=img_size,
         crop_ratio_range=(0.95, 1.0),
         interpolation='bicubic', backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotation90'),
    dict(type='ColorJitter', brightness=0.2, contrast=0.2,
         saturation=0.15, hue=0.04),
    dict(type='PackInputs', algorithm_keys=('defect_type_label',)),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(img_size, img_size),
         interpolation='bicubic', backend='pillow'),
    dict(type='PackInputs'),
]

# ======================== DataLoaders ===============================
batch_size_per_gpu = 64
num_workers = 8

train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MergedKaputtMultiLabelDataset',
        ann_files=[
            f'{parquet_root}/query-train.parquet',
            f'{parquet_root}/query-validation.parquet',
            f'{parquet_root}/query-test.parquet',
        ],
        data_prefixes=[
            f'{data_root}/data/train/query-data/crop',
            f'{data_root}/data/validation/query-data/crop',
            f'{data_root}/data/test/query-data/crop',
        ],
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=batch_size_per_gpu * 2,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KaputtDataset',
        ann_file=f'{parquet_root}/query-test.parquet',
        data_prefix=dict(
            img_path=f'{data_root}/data/test/query-data/crop'),
        pipeline=val_pipeline,
    ),
)

test_dataloader = val_dataloader

# ======================== Evaluator =================================
val_evaluator = [
    dict(type='Accuracy', topk=(1,)),
    dict(type='BinaryAPMetric'),
]
test_evaluator = val_evaluator

# ======================== Optimizer =================================
optim_wrapper = dict(
    type='AmpOptimWrapper',
    accumulative_counts=1,
    optimizer=dict(
        type='AdamW',
        lr=1e-5,
        weight_decay=0.05,
        betas=(0.9, 0.999),
    ),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=20.0, decay_mult=0.1),
            aux_fc=dict(lr_mult=20.0, decay_mult=0.1),
        ),
    ),
    clip_grad=dict(max_norm=1.0),
)

# ======================== LR Schedule ===============================
epochs = 10
warmup_epochs = 2

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3,
         by_epoch=True, begin=0, end=warmup_epochs),
    dict(type='CosineAnnealingLR', eta_min=1e-7,
         by_epoch=True, begin=warmup_epochs, end=epochs),
]

train_cfg = dict(by_epoch=True, max_epochs=epochs, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# ======================== Hooks =====================================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=2,
        save_best=['binary_ap/AP'],
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [
    dict(
        type='GradualUnfreezeHook',
        grad_checkpointing=True,
        unfreeze_schedule={
            1: 0,
            5: 4,
            8: 8,
        },
    ),
]

# ======================== Visualizer ================================
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
)

# ======================== Environment ===============================
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

randomness = dict(seed=42, deterministic=True)

log_level = 'INFO'
load_from = None
resume = False
