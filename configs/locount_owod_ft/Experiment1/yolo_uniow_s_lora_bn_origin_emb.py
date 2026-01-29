# 配置 2：原始預訓練模型 + embeddings
#   - 預訓練模型: pretrained/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth
#   - embeddings: C:/Users/dachen/YOLO-UniOW/embeddings/uniow-s

_base_ = [('../../third_party/mmyolo/configs/yolov10/'
          'yolov10_s_syncbn_fast_8xb16-500e_coco.py')]

num_classes = _base_.PREV_INTRODUCED_CLS + _base_.CUR_INTRODUCED_CLS + 2
num_training_classes = _base_.PREV_INTRODUCED_CLS + _base_.CUR_INTRODUCED_CLS + 2
max_epochs = 20
base_lr = 1e-3
weight_decay = 0.025
train_batch_size_per_gpu = 32

work_dir = 'work_dirs/locount_owod_origin'

load_from = r'pretrained/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth'

model = dict(
    type='OWODDetector',
    mm_neck=False,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    num_prev_classes=_base_.PREV_INTRODUCED_CLS,
    num_prompts=num_classes,
    freeze_prompt=False,
    embedding_path=r'C:/Users/dachen/YOLO-UniOW/embeddings/uniow-s/{_base_.owod_dataset.lower()}_t{_base_.owod_task}.npy',
    unknown_embedding_path=r'C:/Users/dachen/YOLO-UniOW/embeddings/uniow-s/object.npy',
    anchor_embedding_path=r'C:/Users/dachen/YOLO-UniOW/embeddings/uniow-s/object_tuned.npy',
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
                                    freeze_one2one=True,
                                    freeze_one2many=True,
                                    embed_dims=512,
                                    num_classes=num_training_classes)),
    train_cfg=dict(one2many_assigner=dict(num_classes=num_training_classes),
                   one2one_assigner=dict(num_classes=num_training_classes),
                   anchor_label=dict(iou_threshold=0.5, score_threshold=0.01)),
    test_cfg=dict(unknown_nms=dict(iou_threshold=0.99, score_threshold=0.2)),
)
