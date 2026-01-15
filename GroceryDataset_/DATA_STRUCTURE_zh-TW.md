# 📦 GroceryDataset OWOD 數據結構詳細說明

---

## 目錄
1. [整體數據結構](#整體數據結構)
2. [VOC XML 格式詳解](#voc-xml-格式詳解)
3. [ImageSets 文件格式](#imagesets-文件格式)
4. [OWOD 任務分割](#owod-任務分割)
5. [文件示例](#文件示例)

---

## 整體數據結構

### 處理前（COCO 格式）
```
GroceryDataset_part1/
├── ShelfImages/
│   ├── image_001.JPG
│   ├── image_002.JPG
│   └── ... (354 張圖片)
└── annotations_coco.json  (COCO 格式標註)
```

### 處理後（OWOD/VOC 格式）
```
data/OWOD/
├── JPEGImages/                    # 圖片資料夾
│   └── GroceryOWOD/               # 資料集子資料夾
│       ├── image_001.jpg
│       ├── image_002.jpg
│       └── ... (354 張)
│
├── Annotations/                   # VOC XML 標註資料夾
│   └── GroceryOWOD/               # 資料集子資料夾
│       ├── image_001.xml
│       ├── image_002.xml
│       └── ... (354 個)
│
└── ImageSets/                     # 數據集分割文件資料夾
    └── GroceryOWOD/               # 資料集子資料夾
        ├── t1_train.txt           # Task 1 訓練集 (245 張)
        ├── t1_known.txt           # Task 1 已知類別集 (245 張)
        │
        ├── t2_train.txt           # Task 2 訓練集 (245 張)
        ├── t2_known.txt           # Task 2 已知類別集 (245 張)
        │
        ├── t3_train.txt           # Task 3 訓練集 (245 張)
        ├── t3_known.txt           # Task 3 已知類別集 (245 張)
        │
        ├── t4_train.txt           # Task 4 訓練集 (245 張)
        ├── t4_known.txt           # Task 4 已知類別集 (245 張)
        │
        └── test.txt               # 測試集 (109 張)
```

---

## VOC XML 格式詳解

### XML 文件位置
```
data/OWOD/Annotations/GroceryOWOD/image_001.xml
```

### XML 結構與說明
```xml
<?xml version="1.0" encoding="UTF-8"?>
<annotation>
  <!-- 圖片元數據 -->
  <folder>GroceryOWOD</folder>          <!-- 資料集子資料夾名 -->
  <filename>image_001.jpg</filename>    <!-- 圖片文件名 -->
  <path>/workspace/data/OWOD/JPEGImages/GroceryOWOD/image_001.jpg</path>  <!-- 圖片完整路徑 -->
  
  <!-- 圖片尺寸 -->
  <size>
    <width>1280</width>                <!-- 圖片寬度（像素） -->
    <height>720</height>               <!-- 圖片高度（像素） -->
    <depth>3</depth>                   <!-- 色彩通道數（RGB=3） -->
  </size>
  
  <!-- 數據源信息 -->
  <source>
    <database>GroceryDataset</database> <!-- 原始數據集名稱 -->
    <annotation>COCO</annotation>       <!-- 原始標註格式 -->
  </source>
  
  <!-- 分割信息（OWOD 特定） -->
  <owod_split>
    <t1>train</t1>                     <!-- Task 1 中的角色 -->
                                        <!-- 可選值: train, known, test -->
  </owod_split>
  
  <!-- 物體檢測邊界框 -->
  <object>
    <name>category_0</name>             <!-- 類別名稱 -->
    <difficult>0</difficult>            <!-- 難度標記 (0=正常, 1=困難) -->
    <truncated>0</truncated>            <!-- 截斷標記 (圖片邊界外) -->
    <occluded>0</occluded>              <!-- 遮擋標記 -->
    
    <!-- 邊界框坐標 -->
    <bndbox>
      <xmin>100</xmin>                 <!-- 左上角 X 坐標 -->
      <ymin>50</ymin>                  <!-- 左上角 Y 坐標 -->
      <xmax>350</xmax>                 <!-- 右下角 X 坐標 -->
      <ymax>300</ymax>                 <!-- 右下角 Y 坐標 -->
    </bndbox>
  </object>
  
  <!-- 可能有多個物體 -->
  <object>
    <name>category_1</name>
    <difficult>0</difficult>
    <truncated>0</truncated>
    <occluded>0</occluded>
    <bndbox>
      <xmin>400</xmin>
      <ymin>100</ymin>
      <xmax>500</xmax>
      <ymax>250</ymax>
    </bndbox>
  </object>
</annotation>
```

### XML 關鍵字段說明

| 字段 | 說明 | 示例 |
|------|------|------|
| `filename` | 對應的圖片文件名 | `image_001.jpg` |
| `width` | 圖片寬度 | `1280` |
| `height` | 圖片高度 | `720` |
| `name` | 物體類別 | `category_0` |
| `xmin, ymin` | 邊界框左上角 | `(100, 50)` |
| `xmax, ymax` | 邊界框右下角 | `(350, 300)` |
| `difficult` | 是否是困難樣本 | `0 (否) 或 1 (是)` |

---

## ImageSets 文件格式

### 文件位置
```
data/GroceryOWOD/ImageSets/
```

### 文件類型

#### 1️⃣ 訓練集文件 (tX_train.txt)

**文件名**: `t1_train.txt`, `t2_train.txt`, `t3_train.txt`, `t4_train.txt`

**內容**: 每行一個圖片名稱（不含 .jpg 擴展名）

```
image_001
image_003
image_005
image_008
... (共 245 行)
```

**說明**:
- Task X 的訓練集圖片清單
- 每行一個圖片名稱
- 名稱不含 `.jpg` 擴展名
- YOLO-UniOW 會自動找 `JPEGImages/` 下對應的 `.jpg` 文件

#### 2️⃣ 已知類別集文件 (tX_known.txt)

**文件名**: `t1_known.txt`, `t2_known.txt`, `t3_known.txt`, `t4_known.txt`

**內容**: 該任務中包含已知類別的圖片清單

```
image_001
image_003
image_005
image_008
... (共 245 行)
```

**說明**:
- Task X 訓練時的已知類別圖片集合
- 通常與 `tX_train.txt` 內容相同
- 用於計算 Known mAP 評估指標
- 過濾掉了不包含任何已知類別的圖片

#### 3️⃣ 測試集文件 (test.txt)

**文件名**: `test.txt`

**內容**: 所有測試圖片清單

```
image_011
image_022
image_035
image_050
... (共 109 行)
```

**說明**:
- 所有任務共享的測試集
- 獨立於訓練集，不重疊
- 用於評估模型在未見數據上的性能

---

## OWOD 任務分割

### GroceryDataset 的任務分割設計

```
11 個類別 → 4 個遞進式學習任務

Task 1:
  ├─ 已知類別: category_0, category_1, category_2 (3 個)
  ├─ 訓練圖片: t1_train.txt (245 張)
  └─ 包含這些類別的圖片

Task 2 (Task 1 的基礎上):
  ├─ 新增類別: category_3, category_4, category_5 (3 個)
  ├─ 累計已知: 6 個類別
  ├─ 訓練圖片: t2_train.txt (245 張)
  └─ 包含新類別 + 舊類別的圖片

Task 3 (Task 2 的基礎上):
  ├─ 新增類別: category_6, category_7, category_8 (3 個)
  ├─ 累計已知: 9 個類別
  ├─ 訓練圖片: t3_train.txt (245 張)
  └─ 包含新類別 + 舊類別的圖片

Task 4 (Task 3 的基礎上):
  ├─ 新增類別: category_9, category_10 (2 個)
  ├─ 累計已知: 11 個類別
  ├─ 訓練圖片: t4_train.txt (245 張)
  └─ 所有類別都是已知的
```

### 類別對應關係

| Task | 新增類別 | 已知類別 |
|------|---------|---------|
| T1   | category_0, category_1, category_2 | 3 個 |
| T2   | category_3, category_4, category_5 | 6 個 |
| T3   | category_6, category_7, category_8 | 9 個 |
| T4   | category_9, category_10 | 11 個 |

### 數據分配

| 集合 | Task 1 | Task 2 | Task 3 | Task 4 | 測試 | 總計 |
|------|--------|--------|--------|--------|------|------|
| 訓練 | 245    | 245    | 245    | 245    | -    | 980  |
| 測試 | -      | -      | -      | -      | 109  | 109  |
| **合計** | **245** | **245** | **245** | **245** | **109** | **1089** |

---

## 文件示例

### 範例 1: 完整的 XML 文件

```xml
<?xml version="1.0" encoding="UTF-8"?>
<annotation>
  <folder>GroceryOWOD</folder>
  <filename>image_001.jpg</filename>
  <path>/workspace/data/OWOD/JPEGImages/GroceryOWOD/image_001.jpg</path>
  
  <size>
    <width>1280</width>
    <height>720</height>
    <depth>3</depth>
  </size>
  
  <source>
    <database>GroceryDataset</database>
    <annotation>COCO</annotation>
  </source>
  
  <owod_split>
    <t1>train</t1>
  </owod_split>
  
  <object>
    <name>category_0</name>
    <difficult>0</difficult>
    <truncated>0</truncated>
    <occluded>0</occluded>
    <bndbox>
      <xmin>100</xmin>
      <ymin>50</ymin>
      <xmax>350</xmax>
      <ymax>300</ymax>
    </bndbox>
  </object>
  
  <object>
    <name>category_1</name>
    <difficult>0</difficult>
    <truncated>0</truncated>
    <occluded>0</occluded>
    <bndbox>
      <xmin>400</xmin>
      <ymin>100</ymin>
      <xmax>500</xmax>
      <ymax>250</ymax>
    </bndbox>
  </object>
</annotation>
```

### 範例 2: ImageSets 文件內容

**t1_train.txt** (前 10 行)
```
image_001
image_003
image_005
image_008
image_010
image_012
image_015
image_018
image_020
image_022
```

**test.txt** (前 10 行)
```
image_011
image_022
image_035
image_050
image_067
image_089
image_102
image_125
image_150
image_178
```

---

## 邊界框坐標說明

### 坐標系統

```
(0, 0) ─── xmax ─→
  │
  │
ymax
  │
  ↓

圖片高度: 720 像素
圖片寬度: 1280 像素
```

### 坐標計算示例

```
物體在圖片上的位置:

      100       350
      ┌────────┐
   50 │ object │ 
      │   0    │
  300 └────────┘

坐標: xmin=100, ymin=50, xmax=350, ymax=300
寬度: 350 - 100 = 250 像素
高度: 300 - 50 = 250 像素
```

---

## VOC 格式 vs COCO 格式

### 主要差異

| 特性 | VOC XML | COCO JSON |
|------|---------|-----------|
| 文件格式 | XML（每張圖片一個文件） | JSON（單個文件） |
| 邊界框坐標 | [xmin, ymin, xmax, ymax] | [x, y, width, height] |
| 圖片尺寸信息 | 包含在每個 XML 中 | 在 image 字段中 |
| 類別定義 | 通過物體名稱 | 通過 category ID |
| 數據集拆分 | 通過 ImageSets 文本文件 | 通過 split 字段 |

### 坐標轉換公式

```
VOC 格式  → COCO 格式:
width  = xmax - xmin
height = ymax - ymin
x = xmin
y = ymin

COCO 格式 → VOC 格式:
xmin = x
ymin = y
xmax = x + width
ymax = y + height
```

---

## 數據驗證檢查清單

在使用數據前，請檢查以下項目：

### 文件結構檢查
- [ ] `JPEGImages/` 目錄包含 354 張 `.jpg` 圖片
- [ ] `Annotations/` 目錄包含 354 個 `.xml` 文件
- [ ] 每個 `.xml` 文件對應一個 `.jpg` 圖片

### ImageSets 檢查
- [ ] `t1_train.txt`, `t2_train.txt`, `t3_train.txt`, `t4_train.txt` 存在
- [ ] `t1_known.txt`, `t2_known.txt`, `t3_known.txt`, `t4_known.txt` 存在
- [ ] `test.txt` 存在
- [ ] 訓練集總行數: 245 行 × 4 = 980 行
- [ ] 測試集行數: 109 行

### XML 內容檢查
- [ ] 所有 XML 格式正確，可被解析
- [ ] 邊界框坐標有效：`xmin < xmax` 且 `ymin < ymax`
- [ ] 類別名稱為 `category_0` 到 `category_10`
- [ ] 圖片尺寸為正整數

### 數據一致性檢查
- [ ] `ImageSets/` 中的圖片名稱都存在於 `JPEGImages/` 和 `Annotations/` 中
- [ ] 訓練集和測試集不重疊
- [ ] 所有類別都被正確標記

---

## 常見問題

### Q: 為什麼要使用 VOC XML 格式？
A: VOC XML 格式是標準的物體檢測格式，與 YOLO-UniOW 框架兼容性最好，且支持 OWOD 任務的特定分割需求。

### Q: ImageSets 中的圖片順序重要嗎？
A: 順序不重要，但建議使用數值排序以便管理。

### Q: 可以修改 XML 中的字段嗎？
A: 可以，但必須保持以下字段完整：`filename`, `size`, `object/name`, `bndbox`。

### Q: 為什麼 t1_known.txt 和 t1_train.txt 內容相同？
A: 在 Task 1 中，訓練集中的所有圖片都只包含已知類別。在後續任務中，可能包含新類別和未知物體。

---

## 相關文件

- **[QUICKSTART_OWOD_zh-TW.md](QUICKSTART_OWOD_zh-TW.md)** - OWOD 訓練快速開始指南
- **[OWOD_TRAINING_PLAN_zh-TW.md](OWOD_TRAINING_PLAN_zh-TW.md)** - 完整訓練規劃與配置
- **[prepare_grocery_owod.py](prepare_grocery_owod.py)** - 數據格式轉換腳本

---

## 更新日誌

| 日期 | 版本 | 說明 |
|------|------|------|
| 2025-01-15 | 1.0 | 初版創建 |

