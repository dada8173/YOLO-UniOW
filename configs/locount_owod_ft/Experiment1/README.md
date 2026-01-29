# Experiment1 實驗說明

本資料夾包含兩個 LocountOWOD 訓練實驗的設定檔，分別針對模型修復與原始預訓練模型進行比較：

## 1. yolo_uniow_s_lora_bn_5e-5_repair.py
- 目的：針對模型進行保守修復訓練，防止梯度爆炸並提升穩定性。
- 主要設定：
  - Backbone: frozen_stages=4（完全凍結）
  - Neck: freeze_all=True（完全凍結）
  - bbox_head: freeze_one2one=False, freeze_one2many=False（解凍訓練分支）
  - 學習率：5e-5（極低，保守修復）
  - 權重衰減：0.01（減半，提升適應性）
  - 梯度裁剪：max_norm=1.0（強力穩定訓練）
  - 預訓練模型：best_owod_Both_epoch_20.pth
- 適用於模型出現不穩定或需微調修復時。

## 2. yolo_uniow_s_lora_bn_origin_emb.py
- 目的：直接使用原始預訓練模型與 embeddings 進行訓練，作為 baseline 對照。
- 主要設定：
  - 預訓練模型：pretrained/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth
  - embeddings：C:/Users/dachen/YOLO-UniOW/embeddings/uniow-s
  - Backbone/Neck 皆為完全凍結，bbox_head 也凍結（freeze_one2one=True, freeze_one2many=True）
  - 學習率：1e-3（預設）
  - 權重衰減：0.025（預設）
- 適用於驗證原始模型與 embeddings 的效果。

---

這兩個設定檔可用於比較「保守修復訓練」與「原始預訓練模型」在 LocountOWOD 任務上的表現差異，協助選擇最佳訓練策略。