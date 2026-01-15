# 🎯 GroceryDataset OWOD 訓練完整規劃

基於 YOLO-UniOW 的 OWOD (Open-World Object Detection) 框架訓練 GroceryDataset

---

## 📊 數據集概況

- **名稱**: GroceryDataset (雜貨店貨架產品檢測)
- **圖片數**: 354 張
- **標註數**: 13,184 個
- **類別數**: 11 個 (category_0 ~ category_10)
- **當前格式**: COCO JSON
- **目標格式**: VOC XML (OWOD 要求)

---

## 🎓 OWOD 概念說明

### 什麼是 OWOD?

**Open-World Object Detection (開放世界物體檢測)** 模擬真實世界的學習場景：

1. **增量學習**: 逐步學習新類別（Task 1 → Task 2 → Task 3 → Task 4）
2. **未知物體檢測**: 能識別出"未知"物體（不在已知類別中）
3. **知識保留**: 學習新類別時不忘記舊類別

### GroceryDataset 的 OWOD 設計

將 11 個產品類別分為 4 個學習任務：

```
Task 1: 學習 3 個類別  (category_0, 1, 2)
Task 2: 學習 3 個新類別 (category_3, 4, 5) → 累計 6 個
Task 3: 學習 3 個新類別 (category_6, 7, 8) → 累計 9 個
Task 4: 學習 2 個新類別 (category_9, 10)  → 累計 11 個
```

**類別分割建議** (可根據類別不平衡調整):
```python
grocery_owod_settings = {
    "task_list": [0, 3, 6, 9, 11],  # 4 個任務
    "test_image_set": "test"
}
```

---

## 📁 第一步：數據結構組織

### 目標結構

```
data/
├── OWOD/                              # 現有 OWOD 數據集
│   ├── JPEGImages/
│   │   ├── SOWODB/
│   │   ├── MOWODB/
│   │   └── nuOWODB/
│   ├── Annotations/
│   └── ImageSets/
│
├── GroceryOWOD/                       # 新增：您的數據集
│   ├── JPEGImages/                    # 所有圖片
│   │   ├── C1_P01_N1_S2_1.JPG
│   │   └── ...
│   ├── Annotations/                   # VOC XML 標註
│   │   ├── C1_P01_N1_S2_1.xml
│   │   └── ...
│   └── ImageSets/                     # OWOD 任務分割
│       ├── t1_train.txt               # Task 1 訓練圖片 ID
│       ├── t1_known.txt               # Task 1 已知類別
│       ├── t2_train.txt               # Task 2 訓練圖片 ID
│       ├── t2_known.txt               # Task 2 已知類別
│       ├── t3_train.txt
│       ├── t3_known.txt
│       ├── t4_train.txt
│       ├── t4_known.txt
│       └── test.txt                   # 測試集（所有任務共用）
│
└── texts/
    └── grocery_class_texts.json       # 類別文本描述
```

### ImageSets 文件格式

#### **t1_train.txt** (訓練圖片列表)
```plaintext
C1_P01_N1_S2_1
C1_P01_N2_S2_1
C1_P03_N1_S3_1
...
```
每行一個圖片 ID（不含副檔名）

#### **t1_known.txt** (已知類別列表)
```plaintext
category_0
category_1
category_2
```
Task 1 的 3 個已知類別

#### **t2_known.txt**
```plaintext
category_0
category_1
category_2
category_3
category_4
category_5
```
Task 2 累計 6 個類別

---

## 🛠️ 第二步：數據轉換

### 2.1 運行轉換腳本

創建 `prepare_grocery_owod.py` 腳本（見下方），然後運行：

```bash
cd c:\Users\dachen\YOLO-UniOW\GroceryDataset_

# 轉換數據為 OWOD 格式
python prepare_grocery_owod.py
```

這將：
1. 將 COCO JSON 轉換為 VOC XML
2. 創建 OWOD 任務分割（t1-t4）
3. 生成 ImageSets 文件
4. 複製圖片到目標目錄
5. 創建類別文本描述

### 2.2 驗證轉換結果

```bash
# 檢查文件數量
dir data\GroceryOWOD\JPEGImages\*.JPG | measure-object
dir data\GroceryOWOD\Annotations\*.xml | measure-object

# 查看 ImageSets
type data\GroceryOWOD\ImageSets\t1_train.txt | measure-object -line
type data\GroceryOWOD\ImageSets\t1_known.txt

# 檢查一個 XML 文件
type data\GroceryOWOD\Annotations\C1_P01_N1_S2_1.xml
```

---

## ⚙️ 第三步：配置文件設置

### 3.1 數據集配置

創建 `configs/datasets/grocery_owod_dataset.py`:

```python
# GroceryOWOD settings
owod_settings = {
    "GroceryOWOD": {
        "task_list": [0, 3, 6, 9, 11],  # 4 個任務
        "test_image_set": "test"
    }
}

owod_root = "data/GroceryOWOD"

# 從環境變量讀取配置
owod_dataset = '{{$DATASET:GroceryOWOD}}'
owod_task = {{'$TASK:1'}}
train_image_set = '{{$IMAGESET:train}}'
threshold = {{'$THRESHOLD:0.05'}}
training_strategy = {{'$TRAINING_STRATEGY:0'}}
save_rets = {{'$SAVE:False'}}

class_text_path = f"{owod_root}/ImageSets/t{owod_task}_known.txt"
test_image_set = owod_settings[owod_dataset]['test_image_set']

task_list = owod_settings[owod_dataset]['task_list']
PREV_INTRODUCED_CLS = task_list[owod_task - 1]
CUR_INTRODUCED_CLS = task_list[owod_task] - task_list[owod_task - 1]

# OWOD 配置
owod_cfg = dict(
    split=test_image_set,
    task_num=owod_task,
    PREV_INTRODUCED_CLS=PREV_INTRODUCED_CLS,
    CUR_INTRODUCED_CLS=CUR_INTRODUCED_CLS,
    num_classes=PREV_INTRODUCED_CLS + CUR_INTRODUCED_CLS + 1,
)

# 訓練數據集
grocery_train_dataset = dict(
    type='MultiModalOWDataset',
    dataset=dict(
        type='OWODDataset',
        data_root=owod_root,
        image_set=train_image_set,
        dataset=owod_dataset,
        owod_cfg=owod_cfg,
        training_strategy=training_strategy,
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path=class_text_path,
)

# 驗證數據集
grocery_val_dataset = dict(
    type='MultiModalOWDataset',
    dataset=dict(
        type='OWODDataset',
        data_root=owod_root,
        image_set=test_image_set,
        dataset=owod_dataset,
        owod_cfg=owod_cfg,
        test_mode=True),
    class_text_path=class_text_path,
)

# 評估器
grocery_val_evaluator = dict(
    type='OpenWorldMetric',
    data_root=owod_root,
    dataset_name=owod_dataset,
    threshold=threshold,
    save_rets=save_rets,
    owod_cfg=owod_cfg,
)
```

### 3.2 訓練配置

創建 `configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py`:

```python
_base_ = [
    '../../third_party/mmyolo/configs/yolov10/yolov10_s_syncbn_fast_8xb16-500e_coco.py',
    '../datasets/grocery_owod_dataset.py'
]

custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# 超參數
num_classes = _base_.PREV_INTRODUCED_CLS + _base_.CUR_INTRODUCED_CLS + 2
num_training_classes = _base_.PREV_INTRODUCED_CLS + _base_.CUR_INTRODUCED_CLS + 2
max_epochs = 20
save_epoch_intervals = 5
val_interval = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 1e-3
weight_decay = 0.025
train_batch_size_per_gpu = 8  # 根據 GPU 記憶體調整

load_from = 'pretrained/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth'

# Embedding mask
embedding_mask = (
    [0] * _base_.PREV_INTRODUCED_CLS +    # 凍結舊類別
    [1] * _base_.CUR_INTRODUCED_CLS  +    # 訓練新類別
    [1]                               +    # 訓練 unknown
    [0]                                    # 凍結 anchor
)

# 模型設置
model = dict(
    type='OWODDetector',
    mm_neck=False,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    num_prev_classes=_base_.PREV_INTRODUCED_CLS,
    num_prompts=num_classes,
    freeze_prompt=False,
    embedding_path=f'embeddings/uniow-s/grocery_t{_base_.owod_task}.npy',
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
        frozen_stages=4,
    ),
    neck=dict(freeze_all=True),
    bbox_head=dict(
        type='YOLOv10WorldHead',
        infer_type='one2one',
        head_module=dict(
            type='YOLOv10WorldHeadModule',
            use_bn_head=True,
            freeze_one2one=True,
            freeze_one2many=True,
            embed_dims=text_channels,
            num_classes=num_training_classes
        )
    ),
    train_cfg=dict(
        one2many_assigner=dict(num_classes=num_training_classes),
        one2one_assigner=dict(num_classes=num_training_classes),
        anchor_label=dict(iou_threshold=0.5, score_threshold=0.01)
    ),
    test_cfg=dict(
        unknown_nms=dict(iou_threshold=0.99, score_threshold=0.2)
    ),
)

# 數據加載器
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=0,  # Windows 設為 0
    persistent_workers=False,
    collate_fn=dict(type='yolow_collate'),
    dataset={{_base_.grocery_train_dataset}}
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset={{_base_.grocery_val_dataset}}
)

val_evaluator = {{_base_.grocery_val_evaluator}}

# 訓練配置
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=3,
        save_best='owod/Both',
        rule='greater'
    )
)

# 優化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=weight_decay),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# 學習率調度
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05, 
         begin=10, end=max_epochs, T_max=10, by_epoch=True)
]
```

---

## 🚀 第四步：Embeddings 生成

### 4.1 提取文本特徵

```bash
cd c:\Users\dachen\YOLO-UniOW

# 提取 GroceryDataset 的類別文本特徵
python tools/owod_scripts/extract_text_feats.py ^
    --config configs/pretrain/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py ^
    --ckpt pretrained/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth ^
    --save_path embeddings/uniow-s ^
    --dataset GroceryOWOD
```

這將生成：
- `embeddings/uniow-s/grocery_t1.npy` (3 個類別的 embeddings)
- `embeddings/uniow-s/grocery_t2.npy` (6 個類別)
- `embeddings/uniow-s/grocery_t3.npy` (9 個類別)
- `embeddings/uniow-s/grocery_t4.npy` (11 個類別)

### 4.2 使用現有的 Wildcard Embeddings

由於 YOLO-UniOW 已提供 fine-tuned wildcard embeddings，直接使用：
- `embeddings/uniow-s/object.npy` (unknown 類別)
- `embeddings/uniow-s/object_tuned.npy` (anchor 類別)

---

## 🎓 第五步：OWOD 訓練

### 5.1 訓練 Task 1

```bash
cd c:\Users\dachen\YOLO-UniOW

# Task 1: 學習前 3 個類別
set DATASET=GroceryOWOD
set TASK=1
set THRESHOLD=0.05
set SAVE=False

python tools/train_owod.py ^
    configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py ^
    --amp ^
    --work-dir work_dirs/grocery_owod_task1
```

**Windows 單 GPU 訓練**（推薦）

### 5.2 訓練 Task 2-4 (增量學習)

```bash
# Task 2: 從 Task 1 的最佳模型繼續
set TASK=2
python tools/train_owod.py ^
    configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py ^
    --amp ^
    --work-dir work_dirs/grocery_owod_task2 ^
    --cfg-options load_from=work_dirs/grocery_owod_task1/best_owod_Both_epoch_*.pth

# Task 3
set TASK=3
python tools/train_owod.py ^
    configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py ^
    --amp ^
    --work-dir work_dirs/grocery_owod_task3 ^
    --cfg-options load_from=work_dirs/grocery_owod_task2/best_owod_Both_epoch_*.pth

# Task 4
set TASK=4
python tools/train_owod.py ^
    configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py ^
    --amp ^
    --work-dir work_dirs/grocery_owod_task4 ^
    --cfg-options load_from=work_dirs/grocery_owod_task3/best_owod_Both_epoch_*.pth
```

### 5.3 自動化訓練腳本

創建 `train_grocery_owod.bat`:

```batch
@echo off
cd c:\Users\dachen\YOLO-UniOW

set DATASET=GroceryOWOD
set THRESHOLD=0.05
set SAVE=False

echo ========================================
echo Training Task 1
echo ========================================
set TASK=1
python tools/train_owod.py configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py --amp --work-dir work_dirs/grocery_owod_task1

echo ========================================
echo Training Task 2
echo ========================================
set TASK=2
for /f %%i in ('dir /b work_dirs\grocery_owod_task1\best_owod_Both_epoch_*.pth') do set TASK1_CKPT=%%i
python tools/train_owod.py configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py --amp --work-dir work_dirs/grocery_owod_task2 --cfg-options load_from=work_dirs/grocery_owod_task1/%TASK1_CKPT%

echo ========================================
echo Training Task 3
echo ========================================
set TASK=3
for /f %%i in ('dir /b work_dirs\grocery_owod_task2\best_owod_Both_epoch_*.pth') do set TASK2_CKPT=%%i
python tools/train_owod.py configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py --amp --work-dir work_dirs/grocery_owod_task3 --cfg-options load_from=work_dirs/grocery_owod_task2/%TASK2_CKPT%

echo ========================================
echo Training Task 4
echo ========================================
set TASK=4
for /f %%i in ('dir /b work_dirs\grocery_owod_task3\best_owod_Both_epoch_*.pth') do set TASK3_CKPT=%%i
python tools/train_owod.py configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py --amp --work-dir work_dirs/grocery_owod_task4 --cfg-options load_from=work_dirs/grocery_owod_task3/%TASK3_CKPT%

echo ========================================
echo All tasks completed!
echo ========================================
```

運行：
```bash
train_grocery_owod.bat
```

---

## 📊 第六步：評估

### 6.1 評估單個任務

```bash
set DATASET=GroceryOWOD
set TASK=1
set THRESHOLD=0.05
set SAVE=True

python tools/test.py ^
    configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py ^
    work_dirs/grocery_owod_task1/best_owod_Both_epoch_XX.pth
```

### 6.2 評估所有任務

創建 `eval_grocery_owod.bat`:

```batch
@echo off
set DATASET=GroceryOWOD
set THRESHOLD=0.05
set SAVE=True

for %%t in (1 2 3 4) do (
    echo Evaluating Task %%t
    set TASK=%%t
    for /f %%i in ('dir /b work_dirs\grocery_owod_task%%t\best_owod_Both_epoch_*.pth') do (
        python tools/test.py configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py work_dirs/grocery_owod_task%%t/%%i
    )
)
```

### 6.3 評估指標

OWOD 評估會輸出：

- **mAP (Known)**: 已知類別的檢測精度
- **Unknown Recall**: 未知物體的召回率
- **Wilderness Impact (WI)**: 未知物體對已知類別檢測的影響
- **Both**: 綜合指標 (mAP + Unknown Recall)

---

## 📈 預期結果

根據您的數據集規模（354 張圖片，每個任務約 60-90 張訓練圖片）：

| Task | 已知類別 | 預期 mAP | 預期 Unknown Recall |
|------|---------|----------|-------------------|
| T1   | 3       | 35-50%   | 10-20%            |
| T2   | 6       | 30-45%   | 15-25%            |
| T3   | 9       | 28-40%   | 18-28%            |
| T4   | 11      | 25-38%   | 20-30%            |

**注意**: 數據量較小可能影響性能，建議：
1. 使用數據增強
2. 調整訓練輪數
3. 考慮使用類別平衡策略

---

## 🔧 故障排除

### 問題 1: CUDA Out of Memory
```python
# 在配置文件中
train_batch_size_per_gpu = 4  # 降到 4 或 2
```

### 問題 2: 類別不平衡
```python
# 調整類別分割，平衡每個任務的樣本數
# 例如：將 category_0 (10440 樣本) 單獨作為 Task 1
task_list = [0, 1, 4, 7, 11]
```

### 問題 3: Unknown Recall 太低
```python
# 調整 unknown NMS 閾值
test_cfg=dict(
    unknown_nms=dict(
        iou_threshold=0.95,  # 降低 (原 0.99)
        score_threshold=0.15  # 降低 (原 0.2)
    )
)
```

### 問題 4: 找不到 Embeddings
```bash
# 確認 embeddings 已生成
dir embeddings\uniow-s\grocery_t*.npy
dir embeddings\uniow-s\object*.npy
```

---

## 📋 完整工作流程檢查清單

- [ ] 1. 下載圖片數據 (GroceryDataset_part1, part2)
- [ ] 2. 運行 `prepare_grocery_owod.py` 轉換數據
- [ ] 3. 驗證數據結構（JPEGImages, Annotations, ImageSets）
- [ ] 4. 創建 `configs/datasets/grocery_owod_dataset.py`
- [ ] 5. 創建 `configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py`
- [ ] 6. 生成 embeddings (`extract_text_feats.py`)
- [ ] 7. 訓練 Task 1
- [ ] 8. 訓練 Task 2-4 (增量學習)
- [ ] 9. 評估所有任務
- [ ] 10. 分析結果並調優

---

## 📚 參考資料

- [YOLO-UniOW 論文](https://arxiv.org/abs/2412.20645)
- [OWOD 數據準備](docs/data_zh-TW.md)
- [訓練腳本範例](run_owod.sh)
- [OWOD 評估指標](yolo_world/metrics/owod_metric.py)

---

## 🎉 成功後的下一步

1. **調優超參數**: 學習率、batch size、訓練輪數
2. **數據擴充**: 收集更多數據提升性能
3. **類別描述優化**: 改善類別文本描述以提升 CLIP embeddings
4. **部署應用**: 將訓練好的模型應用到實際貨架檢測場景

Good luck! 🚀
