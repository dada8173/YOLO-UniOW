"""
驗證 LocountOWOD 數據集的完整性
檢查：
  1. 照片是否存在且可讀取
  2. 標註 XML 檔案是否存在且有效
  3. 照片與標註的對應關係
"""

import os
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

def validate_locount_dataset(dataset_root="data/OWOD", dataset_name="LocountOWOD"):
    """驗證 LocountOWOD 數據集"""
    
    print(f"\n{'='*80}")
    print(f"開始驗證 {dataset_name} 數據集")
    print(f"{'='*80}\n")
    
    # 路徑配置
    image_sets_dir = os.path.join(dataset_root, "ImageSets", dataset_name)
    jpeg_dir = os.path.join(dataset_root, "JPEGImages", dataset_name)
    anno_dir = os.path.join(dataset_root, "Annotations", dataset_name)
    
    # 驗證基本目錄
    for path, name in [(image_sets_dir, "ImageSets"), (jpeg_dir, "JPEGImages"), (anno_dir, "Annotations")]:
        if not os.path.exists(path):
            print(f"❌ 錯誤: {name} 目錄不存在: {path}")
            return
        print(f"✓ {name} 目錄存在: {path}")
    
    # 讀取所有 split 檔案
    splits = {}
    split_files = [f for f in os.listdir(image_sets_dir) if f.endswith('.txt')]
    print(f"\n找到 {len(split_files)} 個 split 檔案:\n")
    
    for split_file in sorted(split_files):
        split_path = os.path.join(image_sets_dir, split_file)
        with open(split_path, 'r') as f:
            image_ids = [line.strip() for line in f.readlines() if line.strip()]
        splits[split_file] = image_ids
        print(f"  {split_file}: {len(image_ids)} 張圖片")
    
    # 逐個驗證每個 split
    total_missing_images = 0
    total_missing_annos = 0
    total_corrupt_images = 0
    total_issues = defaultdict(list)
    
    for split_name, image_ids in splits.items():
        print(f"\n{'─'*80}")
        print(f"檢查 {split_name} ({len(image_ids)} 張圖片)")
        print(f"{'─'*80}")
        
        missing_images = []
        missing_annos = []
        corrupt_images = []
        size_mismatch = []
        
        for idx, img_id in enumerate(image_ids):
            # 圖片路徑
            jpg_path = os.path.join(jpeg_dir, f"{img_id}.jpg")
            xml_path = os.path.join(anno_dir, f"{img_id}.xml")
            
            # 檢查圖片存在性
            if not os.path.exists(jpg_path):
                missing_images.append(img_id)
                total_missing_images += 1
            else:
                # 嘗試讀取圖片
                try:
                    img = cv2.imread(jpg_path)
                    if img is None:
                        corrupt_images.append(img_id)
                        total_corrupt_images += 1
                except Exception as e:
                    corrupt_images.append((img_id, str(e)))
                    total_corrupt_images += 1
            
            # 檢查標註存在性
            if not os.path.exists(xml_path):
                missing_annos.append(img_id)
                total_missing_annos += 1
            else:
                # 嘗試解析 XML
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    # 簡單驗證 XML 結構
                    if root.find('filename') is None:
                        print(f"  ⚠ {img_id}: XML 缺少 'filename' 欄位")
                except Exception as e:
                    print(f"  ⚠ {img_id}: XML 解析失敗 - {str(e)}")
            
            # 進度提示
            if (idx + 1) % 1000 == 0:
                print(f"  已檢查 {idx + 1}/{len(image_ids)} 張圖片...")
        
        # 彙總 split 結果
        if missing_images or missing_annos or corrupt_images:
            print(f"\n  {split_name} 問題彙總:")
            if missing_images:
                print(f"    ❌ 缺失圖片: {len(missing_images)} 張")
                print(f"       範例: {missing_images[:3]}")
                total_issues[split_name].extend([(f"missing_img:{img_id}") for img_id in missing_images[:5]])
            if missing_annos:
                print(f"    ❌ 缺失標註: {len(missing_annos)} 個")
                print(f"       範例: {missing_annos[:3]}")
                total_issues[split_name].extend([(f"missing_anno:{img_id}") for img_id in missing_annos[:5]])
            if corrupt_images:
                print(f"    ⚠ 損壞/無法讀取圖片: {len(corrupt_images)} 張")
                print(f"       範例: {corrupt_images[:3]}")
                total_issues[split_name].extend([(f"corrupt_img:{img_id}") for img_id in corrupt_images[:5]])
        else:
            print(f"  ✓ {split_name} 完整無誤 ({len(image_ids)} 張圖片)")
    
    # 全域彙總
    print(f"\n{'='*80}")
    print("全域數據集驗證結果")
    print(f"{'='*80}")
    print(f"✓ Split 檔案數量: {len(splits)}")
    print(f"✓ 總圖片數量: {sum(len(ids) for ids in splits.values())}")
    print(f"❌ 缺失圖片: {total_missing_images}")
    print(f"❌ 缺失標註: {total_missing_annos}")
    print(f"⚠ 損壞/無法讀取圖片: {total_corrupt_images}")
    
    if total_missing_images == 0 and total_missing_annos == 0 and total_corrupt_images == 0:
        print(f"\n✓✓✓ 數據集完整無誤！可以開始訓練 ✓✓✓\n")
    else:
        print(f"\n⚠ 數據集存在問題，需要修復\n")
        if total_issues:
            print("問題列表:")
            for split, issues in total_issues.items():
                if issues:
                    print(f"  {split}:")
                    for issue in issues[:5]:
                        print(f"    - {issue}")

def export_problematic_ids(dataset_root="data/OWOD", dataset_name="LocountOWOD", output_file="problematic_ids.txt"):
    """匯出有問題的圖片 ID 到檔案"""
    
    print(f"\n匯出有問題的圖片 ID...")
    
    jpeg_dir = os.path.join(dataset_root, "JPEGImages", dataset_name)
    anno_dir = os.path.join(dataset_root, "Annotations", dataset_name)
    
    problematic = []
    
    # 列出所有標註檔案
    for xml_file in os.listdir(anno_dir):
        if xml_file.endswith('.xml'):
            img_id = xml_file.replace('.xml', '')
            jpg_path = os.path.join(jpeg_dir, f"{img_id}.jpg")
            
            if not os.path.exists(jpg_path):
                problematic.append(f"{img_id} (缺失圖片)")
            else:
                try:
                    img = cv2.imread(jpg_path)
                    if img is None:
                        problematic.append(f"{img_id} (無法讀取圖片)")
                except:
                    problematic.append(f"{img_id} (讀取失敗)")
    
    if problematic:
        with open(output_file, 'w') as f:
            f.write('\n'.join([pid.split(' ')[0] for pid in problematic]))
        print(f"✓ 匯出 {len(problematic)} 個有問題的圖片 ID 到: {output_file}")
    else:
        print(f"✓ 所有圖片都正常")

if __name__ == "__main__":
    import sys
    
    # 檢查是否指定了其他數據集
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "LocountOWOD"
    
    # 執行驗證
    validate_locount_dataset(dataset_name=dataset_name)
    
    # 可選：匯出有問題的 ID
    # export_problematic_ids(dataset_name=dataset_name)
