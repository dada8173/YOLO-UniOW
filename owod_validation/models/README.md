# 模型權重檔案放置說明

## 配置說明

將已訓練的OWOD模型權重放在此目錄中。

### 推薦放置方式

1. **小模型** (yolo_uniow_s):
   ```
   models/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod.pth
   ```

2. **中等模型** (yolo_uniow_m):
   ```
   models/yolo_uniow_m_lora_bn_1e-3_20e_8gpus_owod.pth
   ```

3. **大模型** (yolo_uniow_l):
   ```
   models/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth
   ```

## 獲取模型權重

可以從以下位置找到預訓練模型：

- `../../pretrained/` - 預訓練模型存放目錄
- `../../work_dirs/` - 訓練輸出的模型權重目錄

## 符號連結 (可選)

你也可以使用符號連結來連接到原始位置：

**Windows PowerShell:**
```powershell
New-Item -ItemType SymbolicLink -Path ".\models\model_name.pth" -Target "..\..\pretrained\model_name.pth"
```

**Linux/Mac:**
```bash
ln -s ../../pretrained/model_name.pth ./models/model_name.pth
```
