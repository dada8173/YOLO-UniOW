#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理单个修复的XML文件 - 方案A
"""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

def read_classes():
    """读取类别列表"""
    classes = []
    csv_path = Path(r"C:\Users\opdad\YOLO-UniOW\Locount\analysis_output\class_map_template.csv")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                class_name = row[0].strip().rstrip(',')
                classes.append(class_name)
    return classes[:140]

def parse_xml(xml_path):
    """解析单个XML文件"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 获得图片文件名
    filename = root.find('filename').text
    image_id = filename.replace('.jpg', '').replace('.JPG', '')
    
    # 获得类别
    image_classes = set()
    for obj in root.findall('object'):
        name_elem = obj.find('name')
        if name_elem is not None:
            class_name = name_elem.text.strip()
            image_classes.add(class_name)
    
    return image_id, image_classes

def main():
    xml_path = Path(r"C:\Users\opdad\YOLO-UniOW\data\OWOD\Annotations\LocountOWOD\test014632.xml")
    output_dir = Path(r"C:\Users\opdad\YOLO-UniOW\data\OWOD\ImageSets\LocountOWOD")
    
    print("读取类别信息...")
    classes_list = read_classes()
    print(f"✓ 读取了 {len(classes_list)} 个类别")
    
    print(f"\n解析 {xml_path.name}...")
    image_id, image_classes = parse_xml(xml_path)
    print(f"✓ 图片ID: {image_id}")
    print(f"✓ 包含的类别: {image_classes}")
    
    # 获得类别索引
    class_indices = set()
    for class_name in image_classes:
        try:
            idx = classes_list.index(class_name)
            class_indices.add(idx)
            print(f"  - {class_name} (索引 {idx})")
        except ValueError:
            print(f"  ⚠️ 未找到: {class_name}")
    
    print(f"\n类别索引: {sorted(class_indices)}")
    
    # 判断应该分配给哪些任务
    tasks = []
    if any(idx in range(0, 35) for idx in class_indices):
        tasks.append('t1')
    if any(idx in range(35, 70) for idx in class_indices):
        tasks.append('t2')
    if any(idx in range(70, 105) for idx in class_indices):
        tasks.append('t3')
    if any(idx in range(105, 140) for idx in class_indices):
        tasks.append('t4')
    
    print(f"\n应分配给: {tasks}")
    
    # 检查是train还是test
    if image_id.startswith('test'):
        print(f"\n这是一个TEST图片")
        # 检查是否已在test.txt中
        test_file = output_dir / "test.txt"
        with open(test_file, 'r', encoding='utf-8') as f:
            existing = set(line.strip() for line in f)
        
        if image_id in existing:
            print(f"✓ {image_id} 已在 test.txt 中")
        else:
            print(f"⚠️ {image_id} 不在 test.txt 中，正在添加...")
            existing.add(image_id)
            with open(test_file, 'w', encoding='utf-8') as f:
                for img_id in sorted(existing):
                    f.write(img_id + '\n')
            print(f"✓ 已添加到 test.txt")
    else:
        print(f"\n这是一个TRAIN图片")
        # 更新相应的train文件
        for task in tasks:
            train_file = output_dir / f"{task}_train.txt"
            with open(train_file, 'r', encoding='utf-8') as f:
                existing = set(line.strip() for line in f)
            
            if image_id in existing:
                print(f"✓ {image_id} 已在 {task}_train.txt 中")
            else:
                print(f"⚠️ {image_id} 不在 {task}_train.txt 中，正在添加...")
                existing.add(image_id)
                with open(train_file, 'w', encoding='utf-8') as f:
                    for img_id in sorted(existing):
                        f.write(img_id + '\n')
                print(f"✓ 已添加到 {task}_train.txt")
    
    print("\n✅ 处理完成！")

if __name__ == "__main__":
    main()
