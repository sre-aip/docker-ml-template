import datetime
import os
import sys

# custom_imports = dict(imports=["mmext.seg"], allow_failed_imports=False)
WANDB_PROJECT_NAME = "cls_item1"
EXP_NAME = os.path.basename(sys.argv[1]).split(".")[0]

USE_WANDB = True
ITERATION_NUM = 30000
VAL_INTERVAL = 500
CHECKPOINT_INTERVAL = 5000
BATCH_SIZE = 16

HEIGHT = 512
WIDTH = 512
RESIZE_BACKEND = "pillow"
SAVE_NUM = 100
DETERMINISTIC = True

WORK_ROOT_DIR = "/home/docker/myproject/outputs"

timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
WORK_DIR = f"{WORK_ROOT_DIR}/{EXP_NAME}/{timestamp}"

# If you want standard test, please manually configure the test dataset
###############################################################################
# configs/_base_/models/*.py

data_preprocessor = dict(
    type="ClsDataPreprocessor",
    num_classes=2,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# [backbone] init_cfg and pretrained cannot be setting at the same time
# pretrained is a deprecated, use "init_cfg" instead
checkpoint = "https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c50_8xb32_in1k_20220214-3343eccd.pth"
model = dict(
    type="ImageClassifier",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="ResNetV1c",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style="pytorch",
        init_cfg=dict(
            type="Pretrained",
            checkpoint=checkpoint,
            prefix="backbone.",
        ),
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=2,
        in_channels=2048,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        topk=(1,),
    ),
)


###############################################################################
# configs/_base_/datasets/*.py


train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="mmext.SaveImg", save_root_dir=WORK_DIR, prefix="vis_1_original", save_num=SAVE_NUM),
    # dict(type="RandomResizedCrop", scale=(HEIGHT, WIDTH), backend=RESIZE_BACKEND),
    #  dict(type="Resize", scale=(WIDTH, HEIGHT), keep_ratio=True, backend=RESIZE_BACKEND),
    dict(
        type="mmseg.RandomResize",
        scale=[(480, 480), (520, 520)],
        ratio_range=None,
        keep_ratio=False,
        backend=RESIZE_BACKEND,
    ),
    dict(type="mmseg.RandomRotFlip", rotate_prob=0, flip_prob=1),
    dict(
        type="mmseg.PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(type="RandomCrop", crop_size=(HEIGHT, WIDTH), pad_if_needed=True),
    dict(type="mmext.SaveImg", save_root_dir=WORK_DIR, prefix="vis_2_before_pack", save_num=SAVE_NUM),
    dict(type="PackInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="ResizeEdge", scale=512, edge="short", backend=RESIZE_BACKEND),
    # dict(type="CenterCrop", crop_size=(WIDTH, HEIGHT)),
    dict(type="Resize", scale=(WIDTH, HEIGHT), keep_ratio=True, backend=RESIZE_BACKEND),
    dict(type="PackInputs"),
]

dataset_type = "ImageNet"
metainfo = {"classes": ("none", "item1")}
train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root="/home/docker/Datasets/private/item1/train/images",
        # split="train",
        pipeline=train_pipeline,
        metainfo=metainfo,
    ),
    collate_fn=dict(type="pseudo_collate"),
)

val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root="/home/docker/Datasets/private/item1/val/images",
        # split="val",
        pipeline=test_pipeline,
        metainfo=metainfo,
    ),
    collate_fn=dict(type="pseudo_collate"),
)
test_dataloader = val_dataloader

val_evaluator = [
    dict(type="Accuracy", topk=(1,)),
    dict(type="SingleLabelMetric"),
    dict(type="ConfusionMatrix"),
]
test_evaluator = val_evaluator


###############################################################################
# configs/_base_/schedules/*.py

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="SGD", lr=5e-3, momentum=0.9, weight_decay=0.0001),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500,
    ),
    dict(
        type="PolyLR",
        eta_min=1e-6,
        power=1.0,
        begin=1500,
        end=ITERATION_NUM,
        by_epoch=False,
    ),
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=ITERATION_NUM, val_interval=VAL_INTERVAL)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=8, enable=False)


###############################################################################
# configs/_base_/default_runtime.py

default_scope = "mmpretrain"

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=CHECKPOINT_INTERVAL,
        max_keep_ckpts=5,
        save_best=["accuracy/top1", "single-label/f1-score"],
        greater_keys=["accuracy/top1", "single-label/f1-score"],
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="VisualizationHook", enable=False),
)

custom_hooks = []

env_cfg = dict(
    cudnn_benchmark=not DETERMINISTIC,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

wandb_settings = dict(
    type="WandbVisBackend",
    define_metric_cfg=[
        # train
        dict(name="loss*", step_metric="iter", summary="min"),
        dict(name="lr", step_metric="iter"),
        dict(name="data_time", step_metric="iter"),
        dict(name="time", step_metric="iter"),
        dict(name="memory", step_metric="iter"),
        # val
        dict(name="accuracy/*", step_metric="iter", summary="max"),
        dict(name="single-label/*", step_metric="iter", summary="max"),
    ],
    init_kwargs=dict(
        project=WANDB_PROJECT_NAME,
        name=EXP_NAME,
        dir="/home/docker",
        config={
            "batch_size": BATCH_SIZE,
            "image_size": (HEIGHT, WIDTH),
            "resize_backend": RESIZE_BACKEND,
            "model": model,
            "train_pipeline": train_pipeline,
            "test_pipeline": test_pipeline,
            # "train_dataloader": train_dataloader,  # should be ignored?
            # "val_dataloader": val_dataloader,  # should be ignored?
            "optim_wrapper": optim_wrapper,
            "param_scheduler": param_scheduler,
            "train_cfg": train_cfg,
            "val_cfg": val_cfg,
        },
    ),
)

vis_backends = [dict(type="LocalVisBackend")]
if USE_WANDB:
    vis_backends.append(wandb_settings)

visualizer = dict(type="UniversalVisualizer", vis_backends=vis_backends)
log_processor = dict(type="LogProcessor", window_size=10, by_epoch=False)

log_level = "DEBUG"
load_from = None
resume = False

# Use CUBLAS_WORKSPACE_CONFIG=:4096:8
randomness = dict(seed=0, deterministic=DETERMINISTIC)

work_dir = WORK_DIR
