#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create ImageSets for Locount OWOD dataset - FIXED VERSION
按照OWOD格式為Locount數據集創建ImageSets文件
"""

import os
import csv
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

def read_class_map(csv_path):
    """讀取class_map_template.csv，獲得類別名稱"""
    classes = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 移除尾部的逗號和空白
            class_name = line.strip().rstrip(',').strip()
            if class_name:
                classes.append(class_name)
    return classes[:140]  # 只取前140個類別

def parse_xml_annotations(xml_path, classes_set):
    """解析XML文件，獲得圖片包含的類別"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 獲得圖片文件名
        filename = None
        filename_elem = root.find('filename')
        if filename_elem is not None:
            filename = filename_elem.text
        
        if not filename:
            return None, []
        
        # 獲得圖片包含的所有類別
        image_classes = []
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is not None:
                class_name = name_elem.text.strip()
                if class_name in classes_set:
                    if class_name not in image_classes:  # 避免重複
                        image_classes.append(class_name)
        
        return filename, image_classes
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None, []

def get_class_index(class_name, classes_list):
    """獲得類別在列表中的索引"""
    try:
        return classes_list.index(class_name)
    except ValueError:
        return -1

def main():
    # 設置路徑
    base_path = Path(r"C:\Users\opdad\YOLO-UniOW")
    csv_path = base_path / "Locount" / "analysis_output" / "class_map_template.csv"
    xml_dir = base_path / "data" / "OWOD" / "Annotations" / "LocountOWOD"
    output_dir = base_path / "data" / "OWOD" / "ImageSets" / "LocountOWOD"
    
    # 創建輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("讀取類別信息...")
    classes_list = read_class_map(csv_path)
    classes_set = set(classes_list)  # 用set進行快速查找
    print(f"✓ 讀取了 {len(classes_list)} 個類別")
    
    # 打印前幾個類別和最後幾個類別用於驗證
    print(f"\n前5個類別: {classes_list[:5]}")
    print(f"第35-40個類別: {classes_list[34:40]}")
    print(f"第70-75個類別: {classes_list[69:75]}")
    print(f"最後5個類別: {classes_list[-5:]}")
    
    # 定義各task的類別索引範圍
    task_ranges = {
        't1': (0, 35),      # 索引0-34，對應第1-35類別
        't2': (35, 70),     # 索引35-69，對應第36-70類別
        't3': (70, 105),    # 索引70-104，對應第71-105類別
        't4': (105, 140),   # 索引105-139，對應第106-140類別
    }
    
    # 初始化存儲
    image_data = {}  # {image_name: [class_indices]}
    test_images = set()
    
    print("\n掃描XML標註文件...")
    xml_files = list(xml_dir.glob("*.xml"))
    print(f"✓ 找到 {len(xml_files)} 個XML文件")
    
    for i, xml_file in enumerate(xml_files):
        if (i + 1) % 5000 == 0:
            print(f"  已處理: {i + 1}/{len(xml_files)}")
        
        filename, image_classes = parse_xml_annotations(xml_file, classes_set)
        if filename:
            # 移除副檔名
            image_id = filename.replace('.jpg', '').replace('.JPG', '')
            
            # 獲得類別的索引
            class_indices = []
            for class_name in image_classes:
                idx = get_class_index(class_name, classes_list)
                if idx >= 0:
                    class_indices.append(idx)
            
            if class_indices:  # 只記錄有有效類別的圖片
                image_data[image_id] = class_indices
            
            # 檢查是否為test圖片
            if image_id.startswith('test'):
                test_images.add(image_id)
    
    print(f"✓ 成功解析 {len(image_data)} 個圖片的標註")
    
    # 驗證：列印幾個圖片的類別信息
    print("\n驗證 - 隨機幾個圖片的類別信息：")
    for i, (image_id, class_indices) in enumerate(list(image_data.items())[:3]):
        class_names = [classes_list[idx] for idx in class_indices]
        print(f"  {image_id}: 類別索引{class_indices} -> {class_names}")
    
    # 創建known文件
    print("\n創建known文件...")
    known_files = {}
    
    # t1_known: 第0-34個類別
    known_files['t1'] = classes_list[0:35]
    
    # t2_known: 第0-69個類別
    known_files['t2'] = classes_list[0:70]
    
    # t3_known: 第0-104個類別
    known_files['t3'] = classes_list[0:105]
    
    # t4_known: 第0-139個類別
    known_files['t4'] = classes_list[0:140]
    
    for task, class_list in known_files.items():
        output_file = output_dir / f"{task}_known.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for class_name in class_list:
                f.write(class_name + '\n')
        print(f"✓ 創建 {task}_known.txt ({len(class_list)} 個類別)")
    
    # 創建train和test文件
    print("\n創建train和test文件...")
    train_files = {
        't1': set(),
        't2': set(),
        't3': set(),
        't4': set(),
    }
    
    # 分配train圖片
    for image_id, class_indices in image_data.items():
        # 只處理train圖片
        if image_id.startswith('test'):
            continue
        
        # t1: 包含0-34任何類別的train圖片
        if any(0 <= idx < 35 for idx in class_indices):
            train_files['t1'].add(image_id)
        
        # t2: 包含35-69任何類別的train圖片
        if any(35 <= idx < 70 for idx in class_indices):
            train_files['t2'].add(image_id)
        
        # t3: 包含70-104任何類別的train圖片
        if any(70 <= idx < 105 for idx in class_indices):
            train_files['t3'].add(image_id)
        
        # t4: 包含105-139任何類別的train圖片
        if any(105 <= idx < 140 for idx in class_indices):
            train_files['t4'].add(image_id)
    
    # 寫入train文件
    for task, image_ids in train_files.items():
        output_file = output_dir / f"{task}_train.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for image_id in sorted(image_ids):
                f.write(image_id + '\n')
        print(f"✓ 創建 {task}_train.txt ({len(image_ids)} 個圖片)")
    
    # 寫入test文件
    test_output = output_dir / "test.txt"
    with open(test_output, 'w', encoding='utf-8') as f:
        for image_id in sorted(test_images):
            f.write(image_id + '\n')
    print(f"✓ 創建 test.txt ({len(test_images)} 個圖片)")
    
    print("\n" + "="*50)
    print("✅ 完成！所有ImageSets文件已創建")
    print("="*50)
    
    # 顯示統計信息
    print(f"\n統計信息：")
    print(f"  總圖片數: {len(image_data) + len(test_images)}")
    print(f"  Train圖片: {len(image_data)}")
    print(f"  Test圖片: {len(test_images)}")
    print(f"\n  t1_train: {len(train_files['t1'])} 圖片")
    print(f"  t2_train: {len(train_files['t2'])} 圖片")
    print(f"  t3_train: {len(train_files['t3'])} 圖片")
    print(f"  t4_train: {len(train_files['t4'])} 圖片")
    
    # 計算只屬於某個task的圖片
    only_t1 = train_files['t1'] - (train_files['t2'] | train_files['t3'] | train_files['t4'])
    only_t2 = train_files['t2'] - (train_files['t1'] | train_files['t3'] | train_files['t4'])
    only_t3 = train_files['t3'] - (train_files['t1'] | train_files['t2'] | train_files['t4'])
    only_t4 = train_files['t4'] - (train_files['t1'] | train_files['t2'] | train_files['t3'])
    
    print(f"\n只含該task類別的圖片數：")
    print(f"  只有t1類別: {len(only_t1)} 圖片")
    print(f"  只有t2新增類別: {len(only_t2)} 圖片")
    print(f"  只有t3新增類別: {len(only_t3)} 圖片")
    print(f"  只有t4新增類別: {len(only_t4)} 圖片")
    
    print(f"\n輸出目錄: {output_dir}")

if __name__ == "__main__":
    main()
