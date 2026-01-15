# 🚀 GroceryDataset OWOD 快速開始

使用 GroceryDataset 訓練 OWOD (Open-World Object Detection) 模型的快速指南。

---

## 📚 已創建的完整文檔

我為您準備了三個關鍵文件：

1. **[OWOD_TRAINING_PLAN_zh-TW.md](OWOD_TRAINING_PLAN_zh-TW.md)** 
   - 完整的 OWOD 訓練規劃
   - OWOD 概念詳細說明
   - 完整的配置文件範例
   - 訓練和評估流程

2. **[prepare_grocery_owod.py](prepare_grocery_owod.py)** 
   - COCO → OWOD 格式轉換腳本
   - 自動創建 OWOD 任務分割
   - 生成所有必需的 ImageSets 文件

3. **[DATA_STRUCTURE_zh-TW.md](DATA_STRUCTURE_zh-TW.md)** 
   - 數據結構詳細說明
   - VOC XML 格式解釋
   - ImageSets 文件格式

---

## 🎯 OWOD 關鍵概念

### 什麼是 OWOD?
**Open-World Object Detection** = 增量學習 + 未知物體檢測

### GroceryDataset 的 OWOD 設計
```
11 個類別 → 4 個學習任務

Task 1: 學習 3 個類別  (category_0, 1, 2)
Task 2: 學習 3 個新類別 (category_3, 4, 5) → 累計 6 個
Task 3: 學習 3 個新類別 (category_6, 7, 8) → 累計 9 個
Task 4: 學習 2 個新類別 (category_9, 10)  → 累計 11 個
```

---

## 📋 快速開始步驟

### ✅ 步驟 1: 等待圖片下載完成
確保 `GroceryDataset_part1/ShelfImages/` 有所有 .JPG 圖片

### ✅ 步驟 2: 轉換為 OWOD 格式
```bash
cd c:\Users\dachen\YOLO-UniOW\GroceryDataset_
python prepare_grocery_owod.py
```

輸出：
```
data/GroceryOWOD/
├── JPEGImages/       (354 張)
├── Annotations/      (354 個 XML)
└── ImageSets/
    ├── t1_train.txt, t1_known.txt
    ├── t2_train.txt, t2_known.txt
    ├── t3_train.txt, t3_known.txt
    ├── t4_train.txt, t4_known.txt
    └── test.txt
```

### ✅ 步驟 3: 創建配置文件

#### 3.1 創建 `configs/datasets/grocery_owod_dataset.py`
參考 [OWOD_TRAINING_PLAN_zh-TW.md](OWOD_TRAINING_PLAN_zh-TW.md) 第 3.1 節

#### 3.2 創建 `configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py`
參考 [OWOD_TRAINING_PLAN_zh-TW.md](OWOD_TRAINING_PLAN_zh-TW.md) 第 3.2 節

### ✅ 步驟 4: 生成 Embeddings
```bash
cd c:\Users\dachen\YOLO-UniOW

python tools/owod_scripts/extract_text_feats.py ^
    --config configs/pretrain/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py ^
    --ckpt pretrained/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth ^
    --save_path embeddings/uniow-s ^
    --dataset GroceryOWOD
```

生成：
- `embeddings/uniow-s/grocery_t1.npy`
- `embeddings/uniow-s/grocery_t2.npy`
- `embeddings/uniow-s/grocery_t3.npy`
- `embeddings/uniow-s/grocery_t4.npy`

### ✅ 步驟 5: 訓練 OWOD

#### Task 1
```bash
set DATASET=GroceryOWOD
set TASK=1
set THRESHOLD=0.05

python tools/train_owod.py ^
    configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py ^
    --amp ^
    --work-dir work_dirs/grocery_owod_task1
```

#### Task 2-4
```bash
# Task 2 從 Task 1 繼續
set TASK=2
python tools/train_owod.py ^
    configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py ^
    --amp ^
    --work-dir work_dirs/grocery_owod_task2 ^
    --cfg-options load_from=work_dirs/grocery_owod_task1/best_owod_Both_epoch_XX.pth

# Task 3、4 依此類推...
```

### ✅ 步驟 6: 評估
```bash
set TASK=1
set SAVE=True
python tools/test.py ^
    configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py ^
    work_dirs/grocery_owod_task1/best_owod_Both_epoch_XX.pth
```

---

## 📊 預期結果

| Task | 已知類別 | 預期 mAP | 預期 Unknown Recall |
|------|---------|----------|-------------------|
| T1   | 3       | 35-50%   | 10-20%            |
| T2   | 6       | 30-45%   | 15-25%            |
| T3   | 9       | 28-40%   | 18-28%            |
| T4   | 11      | 25-38%   | 20-30%            |

---

## 🔧 常見問題

### Q: CUDA Out of Memory
```python
# 配置文件中
train_batch_size_per_gpu = 4  # 降低
```

### Q: 找不到 embeddings
```bash
# 確認生成了
dir embeddings\uniow-s\grocery_t*.npy
```

### Q: Unknown Recall 太低
```python
# 配置文件中調整
test_cfg=dict(
    unknown_nms=dict(
        iou_threshold=0.95,   # 降低
        score_threshold=0.15  # 降低
    )
)
```

---

## 📖 詳細文檔

完整的訓練規劃、配置範例、故障排除，請查看：
**[OWOD_TRAINING_PLAN_zh-TW.md](OWOD_TRAINING_PLAN_zh-TW.md)**

---

## ✅ 檢查清單

- [ ] 圖片下載完成
- [ ] 運行 `prepare_grocery_owod.py` 轉換數據
- [ ] 驗證數據結構
- [ ] 創建數據集配置文件
- [ ] 創建訓練配置文件
- [ ] 生成 embeddings
- [ ] 訓練 Task 1
- [ ] 訓練 Task 2-4
- [ ] 評估所有任務

Good luck! 🎉
