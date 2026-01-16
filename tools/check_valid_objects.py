#!/usr/bin/env python3
"""檢查 GroceryOWOD 訓練集中是否有沒有有效邊框的圖像"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

def check_valid_objects(data_root='data/OWOD', dataset='GroceryOWOD', task=4):
    """檢查訓練集中每個圖像是否有有效的邊框"""
    
    # 讀取訓練集文件列表
    image_set_file = f"{data_root}/ImageSets/{dataset}/t{task}_train.txt"
    
    with open(image_set_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]
    
    annotation_dir = f"{data_root}/Annotations/{dataset}"
    
    # 讀取已知類別列表
    known_classes_file = f"{data_root}/ImageSets/{dataset}/t{task}_known.txt"
    with open(known_classes_file, 'r') as f:
        known_classes = set([line.strip() for line in f.readlines()])
    
    print(f"已知類別: {known_classes}")
    print(f"檢查 {dataset} Task {task} 訓練集...")
    print(f"總圖像數: {len(image_ids)}")
    print("-" * 80)
    
    no_valid_objects = []
    has_unknown = []
    
    for idx, img_id in enumerate(image_ids):
        ann_file = os.path.join(annotation_dir, f"{img_id}.xml")
        
        if not os.path.exists(ann_file):
            continue
        
        try:
            tree = ET.parse(ann_file)
            root = tree.getroot()
            
            objects = root.findall('object')
            valid_count = 0
            unknown_count = 0
            
            for obj in objects:
                class_name = obj.find('name').text
                if class_name in known_classes:
                    valid_count += 1
                else:
                    unknown_count += 1
            
            if valid_count == 0:
                no_valid_objects.append((img_id, len(objects), unknown_count))
            if unknown_count > 0:
                has_unknown.append((img_id, valid_count, unknown_count))
                    
        except Exception as e:
            print(f"錯誤: {img_id} - {e}")
    
    print(f"\n沒有有效邊框（已知類別）的圖像: {len(no_valid_objects)}")
    if no_valid_objects:
        for img_id, total, unknown in no_valid_objects[:20]:
            print(f"  {img_id}: {total}個物體，全是未知類別({unknown}個)")
        if len(no_valid_objects) > 20:
            print(f"  ... 還有 {len(no_valid_objects) - 20} 個")
    
    print(f"\n包含未知類別的圖像: {len(has_unknown)}")
    if has_unknown:
        for img_id, valid, unknown in has_unknown[:10]:
            print(f"  {img_id}: {valid}個已知 + {unknown}個未知")
        if len(has_unknown) > 10:
            print(f"  ... 還有 {len(has_unknown) - 10} 個")

if __name__ == '__main__':
    os.chdir('C:\\Users\\dachen\\YOLO-UniOW')
    check_valid_objects()
