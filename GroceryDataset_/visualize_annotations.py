"""
視覺化 Grocery Dataset 的標註結果
"""
import json
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

def visualize_annotations(coco_file, image_dir, num_samples=5, output_dir="visualizations"):
    """
    視覺化 COCO 格式的標註
    
    Args:
        coco_file: COCO JSON 檔案路徑
        image_dir: 圖片資料夾路徑
        num_samples: 要視覺化的圖片數量
        output_dir: 輸出視覺化結果的資料夾
    """
    # 載入 COCO 標註
    with open(coco_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)
    
    # 建立類別 ID 到名稱的映射
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 建立圖片 ID 到標註的映射
    image_to_anns = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_anns:
            image_to_anns[image_id] = []
        image_to_anns[image_id].append(ann)
    
    # 隨機選擇樣本或取前 num_samples 張
    images = coco_data['images']
    if len(images) > num_samples:
        sample_images = random.sample(images, num_samples)
    else:
        sample_images = images[:num_samples]
    
    # 為每個類別生成隨機顏色
    colors = {}
    for cat_id in categories.keys():
        colors[cat_id] = (random.random(), random.random(), random.random())
    
    # 視覺化每張圖片
    for img_info in sample_images:
        image_id = img_info['id']
        image_name = img_info['file_name']
        image_path = os.path.join(image_dir, image_name)
        
        # 讀取圖片
        try:
            img = Image.open(image_path)
        except Exception as e:
            print(f"無法讀取圖片 {image_path}: {e}")
            continue
        
        # 創建圖形
        fig, ax = plt.subplots(1, figsize=(15, 10))
        ax.imshow(img)
        
        # 繪製標註框
        if image_id in image_to_anns:
            annotations = image_to_anns[image_id]
            
            for ann in annotations:
                bbox = ann['bbox']  # [x, y, width, height]
                category_id = ann['category_id']
                
                # 創建矩形框
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), 
                    bbox[2], 
                    bbox[3],
                    linewidth=2,
                    edgecolor=colors[category_id],
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # 添加類別標籤
                label = f"{categories[category_id]} (ID:{category_id})"
                ax.text(
                    bbox[0], 
                    bbox[1] - 5,
                    label,
                    color='white',
                    fontsize=8,
                    bbox=dict(facecolor=colors[category_id], alpha=0.7, edgecolor='none', pad=1)
                )
        
        ax.axis('off')
        plt.title(f"{image_name} - {len(image_to_anns.get(image_id, []))} 個產品", fontsize=14)
        plt.tight_layout()
        
        # 儲存圖片
        output_path = os.path.join(output_dir, f"vis_{image_name}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"已儲存: {output_path}")
        plt.close()
    
    print(f"\n視覺化完成！共生成 {len(sample_images)} 張圖片")
    print(f"結果儲存在: {output_dir}/")

def print_statistics(coco_file):
    """
    印出數據集統計資訊
    """
    with open(coco_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 統計每個類別的數量
    category_counts = {}
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
    
    print("\n=== 數據集統計 ===")
    print(f"總圖片數: {len(coco_data['images'])}")
    print(f"總標註數: {len(coco_data['annotations'])}")
    print(f"類別數: {len(coco_data['categories'])}")
    print("\n每個類別的標註數量:")
    
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    for cat_id in sorted(category_counts.keys()):
        print(f"  {categories[cat_id]} (ID:{cat_id}): {category_counts[cat_id]}")

if __name__ == "__main__":
    coco_file = "annotations_coco.json"
    image_dir = "ShelfImages"
    
    # 印出統計資訊
    print_statistics(coco_file)
    
    # 視覺化標註（預設 5 張圖片，可修改 num_samples 參數）
    print("\n開始視覺化標註...")
    visualize_annotations(coco_file, image_dir, num_samples=5)
