_base_ = [('../../third_party/mmyolo/configs/yolov10/'
          'yolov10_s_syncbn_fast_8xb16-500e_coco.py'),
          '../datasets/owod_dataset.py']
custom_imports = dict(imports=['yolo_world'],
                      allow_failed_imports=False)

# Task 1 Configuration - MEDIUM UNFROZEN (36% trainable parameters)
# 中等解凍：36% 可訓練參數
# 適合：平衡新域適應和COCO知識保留

num_classes = _base_.PREV_INTRODUCED_CLS + _base_.CUR_INTRODUCED_CLS + 2
num_training_classes = _base_.PREV_INTRODUCED_CLS + _base_.CUR_INTRODUCED_CLS + 2
max_epochs = 20
close_mosaic_epochs = max_epochs
save_epoch_intervals = 5
val_interval = 5
val_interval_stage2 = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]

# 中等解凍使用較低學習率（降低5倍防止梯度爆炸）
base_lr = 1e-5
weight_decay = 0.01
train_batch_size_per_gpu = 32

# 輸出文件夾標註解凍程度
work_dir = 'work_dirs/locount_owod/t1_medium_unfrozen_36percent'

import os
# 使用指定的權重
load_from = r'pretrained/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth'

# Override dataset to LocountOWOD - Task 1
_dataset_env = 'LocountOWOD'
if _dataset_env:
    import sys
    sys.modules['__main__']._dataset = _dataset_env

# trainable (1), frozen (0)
embedding_mask = ([0] * _base_.PREV_INTRODUCED_CLS +    # previous classes
                  [1] * _base_.CUR_INTRODUCED_CLS  +    # current class
                  [1]                              +    # unknown class
                  [0])                                  # anchor class

infer_type = "one2one"

# 中等解凍配置：解凍Backbone最後2層、Neck、One2Many，36% 參數可訓練
model = dict(
    type='OWODDetector',
    mm_neck=False,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    num_prev_classes=_base_.PREV_INTRODUCED_CLS,
    num_prompts=num_classes,
    freeze_prompt=False,
    embedding_path='embeddings/uniow-s/locountowod_t1.npy',
    unknown_embedding_path='embeddings/uniow-s/object.npy',
    anchor_embedding_path='embeddings/uniow-s/object_tuned.npy',
    embedding_mask=embedding_mask,
    data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=None,
        with_text_model=False,
        frozen_stages=3,  # ✅ 解凍最後2層 Stage 3,4 (36%)
    ),
    neck=dict(
        freeze_all=False,  # ✅ 解凍
    ),
    bbox_head=dict(type='YOLOv10WorldHead',
                   infer_type=infer_type,
                   head_module=dict(type='YOLOv10WorldHeadModule',
                                    use_bn_head=True,
                                    freeze_one2one=True,  # 🔒 保留推理精度
                                    freeze_one2many=False,  # ✅ 解凍訓練分支
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(one2many_assigner=dict(num_classes=num_training_classes),
                   one2one_assigner=dict(num_classes=num_training_classes),
                   anchor_label=dict(iou_threshold=0.5, score_threshold=0.01)),
    test_cfg=dict(unknown_nms=dict(iou_threshold=0.99, score_threshold=0.2)),
)

# dataset settings
owod_train_dataset = dict(
    _delete_=True,
    type='MultiModalOWDataset',
    dataset=dict(
        type='OWODDataset',
        data_root=_base_.owod_root,
        image_set=_base_.train_image_set,
        dataset=_base_.owod_dataset,
        owod_cfg=_base_.owod_cfg,
        training_strategy=_base_.training_strategy,
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path=_base_.class_text_path,
    pipeline=_base_.train_pipeline
)

train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=owod_train_dataset)

owod_val_dataset = dict(
    _delete_=True,
    **_base_.owod_val_dataset,
    pipeline=_base_.test_pipeline
)

val_dataloader = dict(batch_size=32, dataset=owod_val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    **_base_.owod_val_evaluator,
)
test_evaluator = val_evaluator

# training settings
default_hooks = dict(param_scheduler=dict(max_epochs=max_epochs),
                     checkpoint=dict(interval=save_epoch_intervals,
                                     save_best=['owod/Both'],
                                     rule='greater'))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=_base_.train_pipeline_stage2)
]

train_cfg = dict(max_epochs=max_epochs,
                 val_interval=val_interval,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     val_interval_stage2)])

optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     clip_grad=dict(max_norm=10.0, norm_type=2),
                     paramwise_cfg=dict(bias_decay_mult=0.0,
                                        norm_decay_mult=0.0,
                                        custom_keys={
                                            'backbone.text_model':
                                            dict(lr_mult=1),
                                            'logit_scale':
                                            dict(weight_decay=0.0),
                                            'embeddings':
                                            dict(weight_decay=0.0)
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')
