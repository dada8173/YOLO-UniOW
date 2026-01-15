# GroceryOWOD Dataset

## OWOD 任務配置

```python
grocery_owod_settings = {
    "task_list": [0, 3, 6, 9, 11],
    "test_image_set": "test"
}
```

## 任務詳情

### Task 1
- **新增類別**: 3 個
- **累計類別**: 3 個
- **類別列表**: ['category_3', 'category_10', 'category_9']

### Task 2
- **新增類別**: 3 個
- **累計類別**: 6 個
- **類別列表**: ['category_5', 'category_6', 'category_8']

### Task 3
- **新增類別**: 3 個
- **累計類別**: 9 個
- **類別列表**: ['category_1', 'category_7', 'category_4']

### Task 4
- **新增類別**: 2 個
- **累計類別**: 11 個
- **類別列表**: ['category_2', 'category_0']


## 文件結構 (OWOD 標準格式)

```
data/OWOD/
├── JPEGImages/GroceryOWOD/       (354 張 .jpg 圖片)
├── Annotations/GroceryOWOD/      (354 個 .xml 文件)
└── ImageSets/GroceryOWOD/
    ├── t1_train.txt
    ├── t1_known.txt
    ├── t2_train.txt
    ├── t2_known.txt
    ├── t3_train.txt
    ├── t3_known.txt
    ├── t4_train.txt
    ├── t4_known.txt
    └── test.txt
```

## 數據統計

- **總圖片數**: 354
- **訓練集**: 247 張
- **測試集**: 107 張
- **總類別數**: 11

## 下一步

1. 生成 embeddings:
```bash
python tools/owod_scripts/extract_text_feats.py \
    --config configs/pretrain/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py \
    --ckpt pretrained/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth \
    --save_path embeddings/uniow-s \
    --dataset GroceryOWOD
```

2. 訓練 Task 1:
```bash
set DATASET=GroceryOWOD
set TASK=1
python tools/train_owod.py configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py --amp
```
