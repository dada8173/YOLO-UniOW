# YOLO-UniOW：高效通用開放世界物體檢測

**YOLO-UniOW** 的官方實現 [[`arxiv`](https://arxiv.org/abs/2412.20645)]

![yolo-uniow](./assets/yolo-uniow.jpg)


## LVIS 數據集上的零樣本性能

YOLO-UniOW-S/M/L 已從頭開始預訓練並在 `LVIS minival` 上進行評估。預訓練的權重可從下方連結下載。

|                            模型                             | 參數數量 | AP<sup>mini</sup> | AP<sub>r</sub> | AP<sub>c</sub> | AP<sub>f</sub> | FPS (V100) |
| :----------------------------------------------------------: | :-----: | :------------------: | :-------------: | :-------------: | :-------------: | :--------: |
| [YOLO-UniOW-S](https://huggingface.co/leonnil/yolo-uniow/resolve/main/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth) |  7.5M   |         26.2         |      24.1       |      24.9       |      27.7       |    98.3    |
| [YOLO-UniOW-M](https://huggingface.co/leonnil/yolo-uniow/resolve/main/yolo_uniow_m_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth) |  16.2M  |         31.8         |      26.0       |      30.5       |       34        |    86.2    |
| [YOLO-UniOW-L](https://huggingface.co/leonnil/yolo-uniow/resolve/main/yolo_uniow_l_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth) |  29.4M  |         34.6         |      30.0       |      33.6       |      36.3       |    64.8    |

## 實驗設置

### 資料準備

有關準備開放詞彙和開放世界數據，請參閱 [docs/data](./docs/data.md)。

### 安裝

我們的模型使用 **CUDA 11.8** 和 **PyTorch 2.1.2** 構建。若要設置環境，請參閱 [PyTorch 官方文檔](https://pytorch.org/get-started/locally/)以獲取安裝指南。有關安裝 `mmcv` 的詳細說明，請參閱 [docs/installation](./docs/installation.md)。

```bash
conda create -n yolouniow python=3.9
conda activate yolouniow
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install -r requirements.txt
pip install -e .
```

### 訓練和評估

有關開放詞彙模型的訓練和評估，請參閱 `run_ovod.sh`

```bash
# 訓練開放詞彙模型
./tools/dist_train.sh configs/pretrain/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py 8 --amp

# 評估開放詞彙模型
./tools/dist_test.sh configs/pretrain/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py \
    pretrained/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth 8
```

有關開放世界模型的訓練和評估，請遵循 `run_owod.sh` 中提供的步驟。在進行評估之前，請確保模型已訓練。我們提供通過步驟 2 和 3 獲得的微調通配符特徵，[object_tuned_s](https://huggingface.co/leonnil/yolo-uniow/resolve/main/object_tuned_s.npy) 和 [object_tuned_m](https://huggingface.co/leonnil/yolo-uniow/resolve/main/object_tuned_m.npy)，允許直接使用。

```bash
# 1. 從預訓練模型提取文本/通配符特徵
python tools/owod_scripts/extract_text_feats.py --config $CONFIG --ckpt $CHECKPOINT --save_path $EMBEDS_PATH

# 2. 微調通配符特徵
./tools/dist_train.sh $OBJ_CONFIG 8 --amp

# 3. 提取微調的通配符特徵
python tools/owod_scripts/extract_text_feats.py --config $OBJ_CONFIG --save_path $EMBEDS_PATH --extract_tuned

# 4. 訓練所有開放世界檢測任務
python tools/owod_scripts/train_owod_tasks.py MOWODB $OW_CONFIG $CHECKPOINT

# 5. 評估所有開放世界檢測任務
python tools/owod_scripts/test_owod_tasks.py MOWODB $OW_CONFIG --save
```

若要在特定資料集和任務上進行訓練和評估，請使用以下命令：

```bash
# 訓練開放世界檢測任務
DATASET=$DATASET TASK=$TASK THRESHOLD=$THRESHOLD SAVE=$SAVE \
./tools/dist_train_owod.sh $CONFIG 8 --amp

# 評估開放世界檢測任務
DATASET=$DATASET TASK=$TASK THRESHOLD=$THRESHOLD SAVE=$SAVE \
./tools/dist_test.sh $CONFIG $CHECKPOINT 8
```

## 致謝

本項目建立在 [YOLO-World](https://github.com/AILab-CVC/YOLO-World)、[YOLOv10](https://github.com/Trami1995/YOLOv10)、[FOMO](https://github.com/orrzohar/FOMO) 和 [OVOW](https://github.com/343gltysprk/ovow/) 的基礎上。我們衷心感謝這些作者的出色實現！

## 引用

如果我們的代碼或模型對您的工作有幫助，請引用我們的論文和 YOLOv10：

```bash
@article{liu2024yolouniow,
  title={YOLO-UniOW: Efficient Universal Open-World Object Detection},
  author={Liu, Lihao and Feng, Juexiao and Chen, Hui and Wang, Ao and Song, Lin and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2412.20645},
  year={2024}
}

@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}
```
