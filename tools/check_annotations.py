#!/usr/bin/env python3
"""檢查 GroceryOWOD 資料集的標註問題"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

def check_annotations(data_root='data/OWOD', dataset='GroceryOWOD', task=4):
    """檢查資料集的標註文件"""
    
    # 讀取訓練集文件列表
    image_set_file = f"{data_root}/ImageSets/{dataset}/t{task}_train.txt"
    
    with open(image_set_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]
    
    annotation_dir = f"{data_root}/Annotations/{dataset}"
    
    print(f"檢查 {dataset} Task {task} 訓練集...")
    print(f"總圖像數: {len(image_ids)}")
    print("-" * 80)
    
    problematic_images = []
    empty_bbox_images = []
    
    for idx, img_id in enumerate(image_ids):
        ann_file = os.path.join(annotation_dir, f"{img_id}.xml")
        
        # 檢查文件是否存在
        if not os.path.exists(ann_file):
            problematic_images.append((img_id, "標註文件不存在"))
            continue
        
        # 檢查 XML 格式和邊框
        try:
            tree = ET.parse(ann_file)
            root = tree.getroot()
            
            objects = root.findall('object')
            if len(objects) == 0:
                empty_bbox_images.append(img_id)
                
        except Exception as e:
            problematic_images.append((img_id, f"XML解析錯誤: {str(e)}"))
    
    print(f"\n問題圖像數: {len(problematic_images)}")
    if problematic_images:
        print("問題圖像:")
        for img_id, error in problematic_images[:20]:
            print(f"  {img_id}: {error}")
        if len(problematic_images) > 20:
            print(f"  ... 還有 {len(problematic_images) - 20} 個")
    
    print(f"\n沒有邊框標註的圖像: {len(empty_bbox_images)}")
    if empty_bbox_images:
        print("這些圖像無邊框:")
        for img_id in empty_bbox_images[:20]:
            print(f"  {img_id}")
        if len(empty_bbox_images) > 20:
            print(f"  ... 還有 {len(empty_bbox_images) - 20} 個")
    
    # 統計信息
    valid_images = len(image_ids) - len(problematic_images) - len(empty_bbox_images)
    print(f"\n統計:")
    print(f"  有效圖像 (有邊框): {valid_images}")
    print(f"  無邊框圖像: {len(empty_bbox_images)}")
    print(f"  問題圖像: {len(problematic_images)}")
    
    return {
        'total': len(image_ids),
        'valid': valid_images,
        'empty': len(empty_bbox_images),
        'problematic': len(problematic_images),
        'empty_images': empty_bbox_images,
        'problem_images': problematic_images
    }

if __name__ == '__main__':
    os.chdir('/Users/dachen/YOLO-UniOW')
    result = check_annotations()
