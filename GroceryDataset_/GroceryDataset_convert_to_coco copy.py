"""
將 Grocery Dataset 的標註格式轉換為 COCO 格式
"""
import json
import os
from pathlib import Path
from PIL import Image

def convert_to_coco(annotation_file, image_dir, output_file):
    """
    將 annotation.txt 轉換為 COCO JSON 格式
    
    Args:
        annotation_file: annotation.txt 的路徑
        image_dir: 圖片資料夾路徑
        output_file: 輸出的 COCO JSON 檔案路徑
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 定義類別（根據 README 有 10 個產品類別，編號 0-10）
    # 類別 0 可能是背景或其他，1-10 是產品類別
    category_names = {
        0: "category_0",
        1: "category_1",
        2: "category_2",
        3: "category_3",
        4: "category_4",
        5: "category_5",
        6: "category_6",
        7: "category_7",
        8: "category_8",
        9: "category_9",
        10: "category_10"
    }
    
    for cat_id, cat_name in category_names.items():
        coco_format["categories"].append({
            "id": cat_id,
            "name": cat_name,
            "supercategory": "product"
        })
    
    image_id = 1
    annotation_id = 1
    
    # 讀取 annotation.txt
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
                
            image_name = parts[0]
            num_products = int(parts[1])
            
            # 獲取圖片尺寸
            image_path = os.path.join(image_dir, image_name)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"警告: 無法讀取圖片 {image_path}: {e}")
                continue
            
            # 添加圖片資訊
            coco_format["images"].append({
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height
            })
            
            # 解析標註資訊
            # 格式: x y w h brand_id (重複 num_products 次)
            for i in range(num_products):
                base_idx = 2 + i * 5
                if base_idx + 4 >= len(parts):
                    break
                    
                x = int(parts[base_idx])
                y = int(parts[base_idx + 1])
                w = int(parts[base_idx + 2])
                h = int(parts[base_idx + 3])
                brand_id = int(parts[base_idx + 4])
                
                # COCO 格式使用 [x, y, width, height]
                bbox = [x, y, w, h]
                area = w * h
                
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": brand_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": []  # 此數據集沒有分割資訊
                })
                
                annotation_id += 1
            
            image_id += 1
    
    # 儲存為 JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=2, ensure_ascii=False)
    
    print(f"轉換完成！")
    print(f"- 圖片數量: {len(coco_format['images'])}")
    print(f"- 標註數量: {len(coco_format['annotations'])}")
    print(f"- 類別數量: {len(coco_format['categories'])}")
    print(f"- 輸出檔案: {output_file}")

if __name__ == "__main__":
    # 設定路徑
    annotation_file = "annotation.txt"
    image_dir = "GroceryDataset_part1/ShelfImages"
    output_file = "annotations_coco.json"
    
    convert_to_coco(annotation_file, image_dir, output_file)
