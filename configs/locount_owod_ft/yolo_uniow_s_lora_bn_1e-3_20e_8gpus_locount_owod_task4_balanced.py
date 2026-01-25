_base_ = ['./yolo_uniow_s_lora_bn_1e-3_20e_8gpus_locount_owod.py']

oversample_thr = 0.1

owod_train_dataset = dict(
    _delete_=True,
    type='MultiModalOWDataset',
    dataset=dict(
        type='ClassBalancedDataset',
        oversample_thr=oversample_thr,
        dataset=dict(
            type='OWODDataset',
            data_root=_base_.owod_root,
            image_set=_base_.train_image_set,
            dataset=_base_.owod_dataset,
            owod_cfg=_base_.owod_cfg,
            training_strategy=_base_.training_strategy,
            filter_cfg=dict(filter_empty_gt=True, min_size=32)),
    ),
    class_text_path=_base_.class_text_path,
    pipeline=_base_.train_pipeline,
)

train_dataloader = dict(dataset=owod_train_dataset)
