# OWOD 模型驗證可視化

這個目錄包含用於OWOD (Open-World Object Detection) 模型的推断和可視化工具。

## 目錄結構

```
owod_validation/
├── images/          # 放置要測試的照片
├── models/          # 放置模型權重檔案
├── outputs/         # 輸出的檢測結果圖片
└── owod_inference.py # 推断和可視化腳本
```

## 使用方法

### 1. 準備檔案

- 將要測試的圖片放在 `images/` 目錄下 (支持 .jpg, .png, .jpeg)
- 將模型權重放在 `models/` 目錄下

### 2. 運行推断

```bash
python owod_inference.py \
    --config <config_file_path> \
    --checkpoint <checkpoint_file_path> \
    --images ./images \
    --output ./outputs \
    --device cuda:0 \
    --score-thr 0.3
```

### 3. 參數說明

- `--config`: 模型配置文件路徑（必需）
- `--checkpoint`: 模型權重檔案路徑（必需）
- `--images`: 輸入圖片目錄或單個圖片路徑（默認: ./images）
- `--output`: 輸出結果目錄（默認: ./outputs）
- `--device`: 推断使用的設備（默認: cuda:0）
- `--score-thr`: 置信度閾值（默認: 0.3）
- `--max-dets`: 最多檢測數量（默認: 100）
- `--show`: 是否顯示結果圖片（可選）

### 4. 範例

#### 範例1: 測試單張圖片
```bash
python owod_inference.py \
    --config ../configs/owod_ft/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod.py \
    --checkpoint ./models/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod.pth \
    --images ./images/test_image.jpg \
    --output ./outputs
```

#### 範例2: 測試整個目錄
```bash
python owod_inference.py \
    --config ../configs/owod_ft/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod.py \
    --checkpoint ./models/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod.pth \
    --images ./images \
    --output ./outputs \
    --score-thr 0.35
```

#### 範例3: 使用CPU推断
```bash
python owod_inference.py \
    --config ../configs/owod_ft/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod.py \
    --checkpoint ./models/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod.pth \
    --images ./images \
    --output ./outputs \
    --device cpu
```

## 輸出

- 檢測結果將以 `<original_name>_result.jpg` 的格式保存在 `outputs/` 目錄下
- 每個圖片上都會繪製檢測框和置信度分數

## 注意事項

1. 確保模型權重檔案與配置文件匹配
2. 如果使用GPU，確保有足夠的顯存
3. 輸入圖片的解析度不要太大（建議 <= 1280x1280）
