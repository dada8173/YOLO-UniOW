## 資料準備

### 資料集目錄

所有資料放在 `data` 目錄，例如：

```bash
├── coco
│   ├── annotations
│   ├── lvis
│   ├── train2017
│   ├── val2017
├── flickr
│   ├── annotations
│   └── full_images
├── mixed_grounding
│   ├── annotations
│   ├── images
├── objects365v1
│   ├── annotations
│   ├── train
│   ├── val
├── OWOD
│   ├── JPEGImages
│   ├── Annotations
│   └── ImageSets
```
**注意：** 強烈建議檢查配置文件中資料集路徑，尤其是 `ann_file`、`data_root`、`data_prefix`。

### 開放詞彙資料集

YOLO-UniOW 預訓練使用下列資料集：

| Data | Samples | Type | Boxes |
| :-- | :-----: | :---:| :---: |
| Objects365v1 | 609k | detection | 9,621k |
| GQA | 621k | grounding | 3,681k |
| Flickr | 149k | grounding | 641k |

我們提供的預訓練標註：

| Data | images | Annotation File |
| :--- | :------| :-------------- |
| Objects365v1 | [Objects365 train](https://opendatalab.com/OpenDataLab/Objects365_v1) | [objects365_train.json](https://opendatalab.com/OpenDataLab/Objects365_v1) |
| MixedGrounding | [GQA](https://nlp.stanford.edu/data/gqa/images.zip) | [final_mixed_train_no_coco.json](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations/final_mixed_train_no_coco.json) |
| Flickr30k | [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) | [final_flickr_separateGT_train.json](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations/final_flickr_separateGT_train.json) |
| LVIS-minival | [COCO val2017](https://cocodataset.org/) | [lvis_v1_minival_inserted_image_name.json](https://huggingface.co/GLIPModel/GLIP/blob/main/lvis_v1_minival_inserted_image_name.json) |

**致謝：** 預訓練資料準備流程基於 [YOLO-World](https://github.com/AILab-CVC/YOLO-World/blob/master/docs/data.md)。

### 開放世界資料集

`data/OWOD` 的結構如下：

```bash
├── OWOD/
│   ├── JPEGImages/
│   │   ├── SOWODB/
│   │   ├── MOWODB/
│   │   └── nuOWODB/
│   ├── Annotations/
│   │   ├── SOWODB/
│   │   ├── MOWODB/
│   │   └── nuOWODB/
│   ├── ImageSets/
│   │   ├── SOWODB/
│   │   ├── MOWODB/
│   │   └── nuOWODB/
```

`data/OWOD/ImageSets/` 下已包含各資料集的切分與已知類別文本提示。

- 下載 [MS-COCO](https://cocodataset.org/#download) 圖像與標註，將 `train2017/`、`val2017/` 的圖像移到 `JPEGImages`。使用 `tools/dataset_converters/coco_to_voc.py` 將 json 轉為 VOC xml。
- 下載 [PASCAL VOC 2007 & 2012](http://host.robots.ox.ac.uk/pascal/VOC/) 圖像與標註。解壓後將圖像放到 `JPEGImages`，標註放到 `Annotations`。
- 下載 [nuImages](https://www.nuscenes.org/nuimages)（用於 nuOWODB），使用 `tools/dataset_converters/nuimages_to_voc.py` 轉換 train/val 標註到 xml，並將圖像與標註放入 `JPEGImages`、`Annotations`，確保檔名與 `ImageSets` 一致。

**注意：** 對於 M-OWODB / S-OWODB，我們將所有 JPEG 圖像與標註放在同一資料夾 `SOWODB`，並對 `MOWODB` 使用符號連結。資料格式沿用 VOC 以便載入與評估。
