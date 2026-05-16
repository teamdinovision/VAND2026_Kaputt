"""Kaputt DINOv2 Defect Binary-Classification -- MMPretrain Config
Integrated tricks

Multi-scale training -> RandomResizedCrop wide ratio
Focal Loss (class imbalance) -> SoftFocalLoss
Gradual Unfreezing -> GradualUnfreezeHook
Mixed Precision + Grad Accum -> AmpOptimWrapper
TTA -> evaluate_vit_mm.py (runtime)
Hard Example Mining (OHEM) -> SoftFocalLoss.ohem_ratio
Mixup / CutMix -> model.train_cfg.augments
Defect-type augmentation -> DefectAwareTransform
Model Ensemble -> evaluate_vit_mm.py (runtime)
Usage:
Single GPU: python train_vit_mm.py configs/kaputt_dinov2.py
4x A100: torchrun --nproc_per_node=4 train_vit_mm.py configs/kaputt_dinov2.py
"""

# ======================== Backbone Selection ========================
# Switch backbone by changing this single variable.
# *_reg4 variants use register tokens -> fewer attention artifacts, better for
# dense tasks like defect detection.
backbone_name = 'dinov3_l'  # 'dinov2_b' | 'dinov2_l' | 'dinov2_l_reg' | 'dinov2_g'

_REGISTRY = dict(
    dinov2_b=dict(timm='vit_base_patch14_dinov2.lvd142m',           ch=768,  ps=14),
    dinov2_l=dict(timm='vit_large_patch14_dinov2.lvd142m',          ch=1024, ps=14),
    dinov2_l_reg=dict(timm='vit_large_patch14_reg4_dinov2.lvd142m', ch=1024, ps=14),
    dinov2_g=dict(timm='vit_giant_patch14_dinov2.lvd142m',          ch=1536, ps=14),
    # --- DINOv3 family (patch_size=16, requires timm >= 1.0.20) ---
    dinov3_b=dict(timm='vit_base_patch16_dinov3.lvd1689m',          ch=768,  ps=16),
    dinov3_l=dict(timm='vit_large_patch16_dinov3.lvd1689m',         ch=1024, ps=16),
)
_bb = _REGISTRY[backbone_name]

# ======================== Image Size ================================
# DINOv2 native pre-training resolution: 518 = 37 * 14
# Higher resolution captures finer defect details; 518 is the minimum
# recommended. Use 728 (52*14) if GPU memory allows.
target_img_size = 728
patch_size = _bb['ps']
img_size = (target_img_size // patch_size) * patch_size  # 518 -> 518

# ======================== Data Paths ================================
data_root = '/data/public/dataset/kaputt'
parquet_root = '/data/public/dataset/kaputt'

# ======================== Runtime ===================================
default_scope = 'mmpretrain'
work_dir = f'./work_dirs/kaputt_{backbone_name}_mm'

# ======================== Model =====================================
model = dict(
    type='ImageClassifier',
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
        type='LinearClsHead',
        num_classes=2,
        in_channels=_bb['ch'],
        loss=dict(
            type='SoftFocalLoss',
            gamma=2.0,
            alpha=0.75,
            ohem_ratio=0.7,
            loss_weight=1.0,
        ),
#        loss=dict(
#             type='AsymmetricFocalLoss',
#             gamma_neg=4,
#             gamma_pos=0,
#             label_smoothing=0.05,
#             loss_weight=1.0,
#         ),
    ),
)

#DDP with gradual unfreezing requires find_unused_parameters
model_wrapper_cfg = dict(find_unused_parameters=True)

# ======================== Data Pipelines ============================#
#--- Trick 1: Multi-scale training (wide crop_ratio_range) ---
#--- Trick 8: Defect-aware augmentation ---
train_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='DefectAwareTransform', extra_prob=0.4, noise_sigma=12.0),
    dict(type='RandomResizedCrop', scale=img_size,
         crop_ratio_range=(0.95, 1.0),
         interpolation='bicubic', backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotation90'),
    dict(type='ColorJitter', brightness=0.2, contrast=0.2,
         saturation=0.15, hue=0.04),
    dict(type='PackInputs'),
]


val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(img_size, img_size),interpolation='bicubic', backend='pillow'),
    dict(type='PackInputs'),
]

# ======================== DataLoaders ===============================
batch_size_per_gpu = 64  # smaller batch -> better generalization; A100-80G
num_workers = 8
##change to train+val
train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MergedKaputtDataset',
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
#        repeat_factors={
#            'book_other': 5,
#            'other': 4,
#            'plastic_tight_wrap': 3,
#        },
        pipeline=train_pipeline,
    ),
)


#train_dataloader = dict(
#    batch_size=batch_size_per_gpu,
#    num_workers=num_workers,
#    persistent_workers=True,
#    pin_memory=True,
#    sampler=dict(type='DefaultSampler', shuffle=True),
#    dataset=dict(
#        type='KaputtDataset',
#        ann_file=f'{parquet_root}/query-train.parquet',
#        data_prefix=dict(
#            img_path=f'{data_root}/data/train/query-data/crop'),
#        pipeline=train_pipeline,
#    ),
#)


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
#
#val_dataloader = dict(
#    batch_size=batch_size_per_gpu * 2,
#    num_workers=num_workers,
#    persistent_workers=True,
#    pin_memory=True,
#    sampler=dict(type='DefaultSampler', shuffle=False),
#    dataset=dict(
#        type='KaputtDataset',
#        ann_file=f'{parquet_root}/query-validation.parquet',
#        data_prefix=dict(
#            img_path=f'{data_root}/data/validation/query-data/crop'),
#        pipeline=val_pipeline,
#    ),
#)

test_dataloader = val_dataloader

# ======================== Evaluator =================================
val_evaluator = [
    dict(type='Accuracy', topk=(1,)),
    dict(type='BinaryAPMetric'),
    dict(type='MaterialAPMetric',
         parquet_path=f'{parquet_root}/query-test.parquet',
         material='plastic_tight_wrap'),
]
test_evaluator = val_evaluator

# ======================== Optimizer =================================
optim_wrapper = dict(
    type='AmpOptimWrapper',
    accumulative_counts=1,  # effective batch = 32 * 4 = 128
    optimizer=dict(
        type='AdamW',
        lr=1e-5,             # lower backbone LR for large model stability
        weight_decay=0.05,
        betas=(0.9, 0.999),
    ),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=20.0, decay_mult=0.1),  # head LR = 2e-4
        ),
    ),
    clip_grad=dict(max_norm=1.0),
)

# ======================== LR Schedule ===============================
epochs = 15
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
        max_keep_ckpts=10,
        save_best=['material_ap/AP', 'binary_ap/AP'],
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# DINOv2-L has 24 blocks; unfreeze more conservatively for stability
custom_hooks = [
    dict(
        type='GradualUnfreezeHook',
        grad_checkpointing=True,
        unfreeze_schedule={
            1: 0, # epoch 1-2 : backbone frozen (head only)
            5: 4, # epoch 3-5 : last 4 blocks unfrozen
            8: 8, # epoch 6-8 : last 8 blocks unfrozen
        },
    ),
    dict(type='SWAHook', swa_start_epoch=15, swa_freq=1),
    dict(type='EarlyStoppingHook', monitor='material_ap/AP',
         patience=15, rule='greater'),
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
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
load_from = None
resume = False