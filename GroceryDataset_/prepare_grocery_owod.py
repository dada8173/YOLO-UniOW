"""
GroceryDataset COCO → OWOD 格式轉換腳本
將 COCO JSON 轉換為 OWOD 所需的 VOC XML 格式，並創建 OWOD 任務分割
"""
import xml.etree.cElementTree as ET
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np


def create_voc_xml(image_info, annotations, categories):
    """創建 VOC 格式 XML"""
    annotation_el = ET.Element('annotation')
    ET.SubElement(annotation_el, 'filename').text = image_info['file_name']
    
    size_el = ET.SubElement(annotation_el, 'size')
    ET.SubElement(size_el, 'width').text = str(image_info['width'])
    ET.SubElement(size_el, 'height').text = str(image_info['height'])
    ET.SubElement(size_el, 'depth').text = str(3)
    
    for ann in annotations:
        object_el = ET.SubElement(annotation_el, 'object')
        category_name = categories[ann['category_id']]
        ET.SubElement(object_el, 'name').text = category_name
        ET.SubElement(object_el, 'difficult').text = '0'
        
        bbox = ann['bbox']
        bb_el = ET.SubElement(object_el, 'bndbox')
        ET.SubElement(bb_el, 'xmin').text = str(int(bbox[0] + 1.0))
        ET.SubElement(bb_el, 'ymin').text = str(int(bbox[1] + 1.0))
        ET.SubElement(bb_el, 'xmax').text = str(int(bbox[0] + bbox[2] + 1.0))
        ET.SubElement(bb_el, 'ymax').text = str(int(bbox[1] + bbox[3] + 1.0))
    
    return ET.ElementTree(annotation_el)


def split_owod_tasks(categories, annotations, task_split=[3, 3, 3, 2]):
    """
    將類別分配到 OWOD 任務中
    
    Args:
        categories: COCO 格式的類別列表 (如 [{'id': 0, 'name': 'category_0'}, ...])
        annotations: 標註列表
        task_split: 每個任務新增的類別數 [T1, T2, T3, T4]
    
    Returns:
        task_categories: {task_id: [category_names]}
        task_images: {task_id: [image_ids]} - 包含該任務類別的圖片
    """
    # 建立類別 id → name 映射
    id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    # 統計每個類別的樣本數
    category_counts = {}
    category_images = {}
    
    for cat_name in id_to_name.values():
        category_counts[cat_name] = 0
        category_images[cat_name] = set()
    
    for ann in annotations:
        cat_name = id_to_name[ann['category_id']]
        category_counts[cat_name] += 1
        category_images[cat_name].add(ann['image_id'])
    
    # 按樣本數排序類別（可選：平衡任務難度）
    sorted_categories = sorted(
        [(name, count) for name, count in category_counts.items()],
        key=lambda x: x[1],
        reverse=False  # False: 從少到多，True: 從多到少
    )
    
    print("\n📊 類別統計（按樣本數排序）:")
    for cat_name, count in sorted_categories:
        print(f"  {cat_name}: {count} 個標註, {len(category_images[cat_name])} 張圖片")
    
    # 分配類別到任務
    task_categories = {}
    task_list = [0]  # [0, 3, 6, 9, 11]
    
    current_idx = 0
    for task_id, num_classes in enumerate(task_split, 1):
        task_cats = []
        for i in range(num_classes):
            if current_idx < len(sorted_categories):
                task_cats.append(sorted_categories[current_idx][0])
                current_idx += 1
        task_categories[task_id] = task_cats
        task_list.append(task_list[-1] + len(task_cats))
        
        print(f"\n✅ Task {task_id}: {task_cats}")
        print(f"   累計類別數: {task_list[-1]}")
    
    print(f"\n📋 task_list = {task_list}")
    
    # 為每個任務收集包含其類別的圖片
    task_images = {task_id: set() for task_id in task_categories.keys()}
    for task_id, cats in task_categories.items():
        for cat_name in cats:
            task_images[task_id].update(category_images[cat_name])
    
    return task_categories, task_images, task_list


def convert_to_owod(coco_file, image_dir, output_root, 
                    dataset_name='GroceryOWOD',
                    task_split=[3, 3, 3, 2], train_ratio=0.7, seed=42):
    """
    完整轉換流程
    
    Args:
        coco_file: COCO 標註文件路徑
        image_dir: 圖片源目錄
        output_root: OWOD 數據集根目錄 (如 data/OWOD)
        dataset_name: 數據集名稱 (如 GroceryOWOD)
        task_split: 每個任務的類別數
        train_ratio: 訓練集比例
        seed: 隨機種子
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print("="*60)
    print("🎯 GroceryDataset → OWOD 格式轉換")
    print("="*60)
    
    # 載入 COCO 數據
    print(f"\n📂 載入 COCO 標註: {coco_file}")
    with open(coco_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    output_root = Path(output_root)
    
    # 創建 OWOD 標準目錄結構
    # data/OWOD/JPEGImages/GroceryOWOD/
    # data/OWOD/Annotations/GroceryOWOD/
    # data/OWOD/ImageSets/GroceryOWOD/
    (output_root / 'JPEGImages' / dataset_name).mkdir(parents=True, exist_ok=True)
    (output_root / 'Annotations' / dataset_name).mkdir(parents=True, exist_ok=True)
    (output_root / 'ImageSets' / dataset_name).mkdir(parents=True, exist_ok=True)
    
    # 類別映射
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_list = [cat['name'] for cat in sorted(coco_data['categories'], key=lambda x: x['id'])]
    
    # 圖片到標註的映射
    image_to_anns = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_anns:
            image_to_anns[image_id] = []
        image_to_anns[image_id].append(ann)
    
    # OWOD 任務分割
    print(f"\n🎓 創建 OWOD 任務分割...")
    task_categories, task_images, task_list = split_owod_tasks(
        coco_data['categories'], coco_data['annotations'], task_split
    )
    
    # 轉換並複製數據
    print(f"\n🔄 轉換 VOC XML 並複製圖片...")
    success_count = 0
    missing_count = 0
    
    all_image_ids = []
    
    for img_info in tqdm(coco_data['images']):
        image_id = img_info['id']
        file_name = img_info['file_name']
        image_name_no_ext = Path(file_name).stem
        
        annotations = image_to_anns.get(image_id, [])
        if not annotations:
            continue
        
        # 創建 XML (注意：OWOD 使用 .jpg 擴展名，不是 .JPG)
        xml_tree = create_voc_xml(img_info, annotations, categories)
        xml_path = output_root / 'Annotations' / dataset_name / f'{image_name_no_ext}.xml'
        xml_tree.write(str(xml_path))
        
        # 複製圖片 (改為 .jpg 擴展名以符合 OWOD 標準)
        possible_paths = [
            Path(image_dir) / file_name,
            Path(image_dir) / Path(file_name).name,
        ]
        
        src_image = None
        for p in possible_paths:
            if p.exists():
                src_image = p
                break
        
        if src_image:
            dst_image = output_root / 'JPEGImages' / dataset_name / f'{image_name_no_ext}.jpg'
            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)
            success_count += 1
            all_image_ids.append((image_name_no_ext, image_id))
        else:
            missing_count += 1
    
    print(f"  ✅ 成功: {success_count} 張")
    if missing_count > 0:
        print(f"  ⚠️  缺失: {missing_count} 張")
    
    # 創建訓練/測試分割
    print(f"\n📑 創建 ImageSets...")
    random.shuffle(all_image_ids)
    n_train = int(len(all_image_ids) * train_ratio)
    
    train_image_ids = set([img_id for _, img_id in all_image_ids[:n_train]])
    test_image_ids = set([img_id for _, img_id in all_image_ids[n_train:]])
    
    print(f"  訓練集: {len(train_image_ids)} 張")
    print(f"  測試集: {len(test_image_ids)} 張")
    
    # 寫入測試集文件
    test_file = output_root / 'ImageSets' / dataset_name / 'test.txt'
    with open(test_file, 'w') as f:
        for img_name, img_id in all_image_ids[n_train:]:
            f.write(f'{img_name}\n')
    print(f"  ✅ {test_file}")
    
    # 為每個任務創建 ImageSets
    for task_id in sorted(task_categories.keys()):
        # 累計所有已學習的類別
        known_categories = []
        for tid in range(1, task_id + 1):
            known_categories.extend(task_categories[tid])
        
        # t{X}_known.txt - 已知類別列表
        known_file = output_root / 'ImageSets' / dataset_name / f't{task_id}_known.txt'
        with open(known_file, 'w') as f:
            for cat_name in known_categories:
                f.write(f'{cat_name}\n')
        
        print(f"\n  Task {task_id}:")
        print(f"    已知類別 ({len(known_categories)}): {known_categories}")
        print(f"    ✅ {known_file}")
        
        # t{X}_train.txt - 訓練圖片列表
        # 只包含當前任務新增類別的圖片
        task_train_images = []
        for img_name, img_id in all_image_ids:
            if img_id not in train_image_ids:
                continue
            
            # 檢查圖片是否包含當前任務的類別
            img_anns = image_to_anns.get(img_id, [])
            img_categories = set([categories[ann['category_id']] for ann in img_anns])
            
            # 包含當前任務任何類別的圖片
            current_task_cats = set(task_categories[task_id])
            if img_categories & current_task_cats:
                task_train_images.append(img_name)
        
        train_file = output_root / 'ImageSets' / dataset_name / f't{task_id}_train.txt'
        with open(train_file, 'w') as f:
            for img_name in task_train_images:
                f.write(f'{img_name}\n')
        
        print(f"    訓練圖片: {len(task_train_images)} 張")
        print(f"    ✅ {train_file}")
    
    # 創建類別文本描述
    print(f"\n📝 創建類別文本描述...")
    texts_dir = Path('../data/texts')
    texts_dir.mkdir(parents=True, exist_ok=True)
    texts_file = texts_dir / 'grocery_class_texts.json'
    
    class_texts = {}
    # 根據實際產品類別自定義描述
    for cat_name in category_list:
        class_texts[cat_name] = [
            cat_name,
            f"a {cat_name}",
            f"a photo of {cat_name}",
            f"{cat_name} on grocery shelf",
            f"grocery product {cat_name}",
        ]
    
    with open(texts_file, 'w', encoding='utf-8') as f:
        json.dump(class_texts, f, indent=2, ensure_ascii=False)
    print(f"  ✅ {texts_file}")
    
    # 創建說明文件
    readme = f"""# GroceryOWOD Dataset

## OWOD 任務配置

```python
grocery_owod_settings = {{
    "task_list": {task_list},
    "test_image_set": "test"
}}
```

## 任務詳情

"""
    
    for task_id, cats in task_categories.items():
        prev_cls = task_list[task_id - 1]
        cur_cls = len(cats)
        total_cls = task_list[task_id]
        
        readme += f"""### Task {task_id}
- **新增類別**: {cur_cls} 個
- **累計類別**: {total_cls} 個
- **類別列表**: {cats}

"""
    
    readme += f"""
## 文件結構 (OWOD 標準格式)

```
data/OWOD/
├── JPEGImages/{dataset_name}/       ({success_count} 張 .jpg 圖片)
├── Annotations/{dataset_name}/      ({success_count} 個 .xml 文件)
└── ImageSets/{dataset_name}/
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

- **總圖片數**: {len(all_image_ids)}
- **訓練集**: {len(train_image_ids)} 張
- **測試集**: {len(test_image_ids)} 張
- **總類別數**: {len(category_list)}

## 下一步

1. 生成 embeddings:
```bash
python tools/owod_scripts/extract_text_feats.py \\
    --config configs/pretrain/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py \\
    --ckpt pretrained/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth \\
    --save_path embeddings/uniow-s \\
    --dataset GroceryOWOD
```

2. 訓練 Task 1:
```bash
set DATASET=GroceryOWOD
set TASK=1
python tools/train_owod.py configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py --amp
```
"""
    
    readme_file = output_root / 'ImageSets' / dataset_name / 'README.md'
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme)
    print(f"  ✅ {readme_file}")
    
    print(f"\n✅ 轉換完成！")
    print(f"📂 輸出目錄: {output_root}")
    print(f"📁 數據集名稱: {dataset_name}")
    
    # 驗證
    verify_owod_data(output_root, dataset_name)
    
    return output_root, dataset_name, task_list


def verify_owod_data(output_root, dataset_name='GroceryOWOD'):
    """驗證 OWOD 數據完整性"""
    print(f"\n🔍 驗證數據完整性...")
    
    output_root = Path(output_root)
    issues = []
    
    # 檢查目錄
    for dir_name in ['JPEGImages', 'Annotations', 'ImageSets']:
        dataset_dir = output_root / dir_name / dataset_name
        if not dataset_dir.exists():
            issues.append(f"❌ 缺少目錄: {dir_name}/{dataset_name}")
        else:
            print(f"  ✅ {dir_name}/{dataset_name}/")
    
    # 檢查文件數量
    image_dir = output_root / 'JPEGImages' / dataset_name
    ann_dir = output_root / 'Annotations' / dataset_name
    
    n_images = len(list(image_dir.glob('*.jpg'))) if image_dir.exists() else 0
    n_xmls = len(list(ann_dir.glob('*.xml'))) if ann_dir.exists() else 0
    
    if n_images != n_xmls:
        issues.append(f"⚠️  圖片數量 ({n_images}) 與標註數量 ({n_xmls}) 不符")
    else:
        print(f"  ✅ 圖片與標註數量一致: {n_images}")
    
    # 檢查 ImageSets
    imageset_dir = output_root / 'ImageSets' / dataset_name
    required_files = []
    for task in [1, 2, 3, 4]:
        required_files.extend([f't{task}_train.txt', f't{task}_known.txt'])
    required_files.append('test.txt')
    
    for file_name in required_files:
        file_path = imageset_dir / file_name
        if file_path.exists():
            with open(file_path, 'r') as f:
                n_lines = len(f.readlines())
            print(f"  ✅ ImageSets/{dataset_name}/{file_name}: {n_lines} 行")
        else:
            issues.append(f"❌ 缺少文件: ImageSets/{dataset_name}/{file_name}")
    
    if issues:
        print("\n⚠️  發現問題:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ 數據完整性檢查通過！")
        return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='將 GroceryDataset 轉換為 OWOD 格式')
    parser.add_argument('--coco-file', type=str, default='annotations_coco.json',
                       help='COCO 標註文件')
    parser.add_argument('--image-dir', type=str, default='GroceryDataset_part1/ShelfImages',
                       help='圖片目錄')
    parser.add_argument('--output-dir', type=str, default='../data/OWOD',
                       help='OWOD 數據集根目錄 (如 data/OWOD)')
    parser.add_argument('--dataset-name', type=str, default='GroceryOWOD',
                       help='數據集名稱子目錄 (如 GroceryOWOD)')
    parser.add_argument('--task-split', type=int, nargs='+', default=[3, 3, 3, 2],
                       help='每個任務的類別數 (例如: 3 3 3 2)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='訓練集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='隨機種子')
    parser.add_argument('--verify-only', action='store_true',
                       help='僅驗證數據')
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_owod_data(args.output_dir, args.dataset_name)
    else:
        convert_to_owod(
            coco_file=args.coco_file,
            image_dir=args.image_dir,
            output_root=args.output_dir,
            dataset_name=args.dataset_name,
            task_split=args.task_split,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
        
        print(f"\n" + "="*60)
        print("🎉 轉換完成！下一步:")
        print("="*60)
        print("1. 創建配置文件:")
        print("   - configs/datasets/grocery_owod_dataset.py")
        print("   - configs/grocery_owod_ft/yolo_uniow_s_grocery_owod.py")
        print("\n2. 生成 embeddings:")
        print("   python tools/owod_scripts/extract_text_feats.py ...")
        print("\n3. 開始訓練:")
        print("   python tools/train_owod.py ...")
        print("\n詳細步驟請參考: OWOD_TRAINING_PLAN_zh-TW.md")
