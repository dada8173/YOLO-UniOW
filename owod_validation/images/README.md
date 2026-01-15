# 測試圖片放置說明

## 使用方法

1. 將要測試的圖片放在此目錄下
2. 支持的格式: `.jpg`, `.png`, `.jpeg`
3. 圖片大小建議: 最大邊長不超過 1280 像素

## 範例

假設你有一張 `test_image.jpg` 的測試圖片，將其放在此目錄：

```
images/
└── test_image.jpg
```

然後運行：
```bash
cd ..
python owod_inference.py \
    --config ../configs/owod_ft/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod.py \
    --checkpoint ./models/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod.pth \
    --images ./images/test_image.jpg \
    --output ./outputs
```

## 建議

- 測試各種不同類型的場景圖片
- 包括室內和室外場景
- 不同光照條件下的圖片
- 物體在不同位置和大小的圖片

## 結果查看

運行完成後，檢測結果會保存在 `../outputs/` 目錄中，文件名為 `<original_name>_result.jpg`
