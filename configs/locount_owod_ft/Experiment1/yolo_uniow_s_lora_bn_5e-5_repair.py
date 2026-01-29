# 配置 1：保守修復訓練
#   - Backbone: frozen_stages=4 (完全凍結)
#   - Neck: freeze_all=True (完全凍結)
#   - bbox_head: freeze_one2one=False, freeze_one2many=False (解凍訓練分支)
#   - 學習率: 5e-5 (保守修復，防止梯度爆炸)
#   - 權重衰減: 0.01 (減半，增加適應性)
#   - 梯度裁剪: max_norm=1.0 (強力穩定訓練)
#   - 預訓練模型: best_owod_Both_epoch_20.pth

_base_ = [('../../third_party/mmyolo/configs/yolov10/'
          'yolov10_s_syncbn_fast_8xb16-500e_coco.py')]

num_classes = _base_.PREV_INTRODUCED_CLS + _base_.CUR_INTRODUCED_CLS + 2
num_training_classes = _base_.PREV_INTRODUCED_CLS + _base_.CUR_INTRODUCED_CLS + 2
max_epochs = 20
base_lr = 5e-5
weight_decay = 0.01
train_batch_size_per_gpu = 32

work_dir = 'work_dirs/locount_owod_repair'

load_from = 'best_owod_Both_epoch_20.pth'

model = dict(
    type='OWODDetector',
    mm_neck=False,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    num_prev_classes=_base_.PREV_INTRODUCED_CLS,
    num_prompts=num_classes,
    freeze_prompt=False,
    embedding_path=f'embeddings/uniow-s/{{_base_.owod_dataset.lower()}}_t{{_base_.owod_task}}.npy',
    unknown_embedding_path='embeddings/uniow-s/object.npy',
    anchor_embedding_path='embeddings/uniow-s/object_tuned.npy',
    embedding_mask=([0] * _base_.PREV_INTRODUCED_CLS + [1] * _base_.CUR_INTRODUCED_CLS + [1] + [0]),
    data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=None,
        with_text_model=False,
        frozen_stages=4,
    ),
    neck=dict(
        freeze_all=True,
    ),
    bbox_head=dict(type='YOLOv10WorldHead',
                   infer_type='one2one',
                   head_module=dict(type='YOLOv10WorldHeadModule',
                                    use_bn_head=True,
                                    freeze_one2one=False,
                                    freeze_one2many=False,
                                    embed_dims=512,
                                    num_classes=num_training_classes)),
    train_cfg=dict(one2many_assigner=dict(num_classes=num_training_classes),
                   one2one_assigner=dict(num_classes=num_training_classes),
                   anchor_label=dict(iou_threshold=0.5, score_threshold=0.01)),
    test_cfg=dict(unknown_nms=dict(iou_threshold=0.99, score_threshold=0.2)),
)

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(bias_decay_mult=0.0,
                       norm_decay_mult=0.0,
                       custom_keys={
                           'backbone.text_model': dict(lr_mult=1),
                           'logit_scale': dict(weight_decay=0.0),
                           'embeddings': dict(weight_decay=0.0)
                       }),
    constructor='YOLOWv5OptimizerConstructor',
    clip_grad=dict(max_norm=1.0, norm_type=2)
)
