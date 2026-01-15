"""
GroceryDataset 標註統計分析與視覺化
生成詳細的統計圖表和報告
"""
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from collections import defaultdict, Counter
import seaborn as sns
import matplotlib.font_manager as fm
import warnings

# 設定中文字體和風格 - 改進版（抑制字體警告）
def setup_chinese_font():
    """設定中文字體，並抑制缺失字符警告"""
    # 抑制matplotlib字體警告
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    # 使用Microsoft YaHei (通常較完整)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 設定字體大小
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    
    print("✓ 字體設定完成 (已抑制字體警告)")

setup_chinese_font()
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_coco_data(coco_file):
    """載入 COCO 格式的標註檔案"""
    with open(coco_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_statistics(coco_data):
    """計算詳細的統計資訊"""
    stats = {
        'total_images': len(coco_data['images']),
        'total_annotations': len(coco_data['annotations']),
        'total_categories': len(coco_data['categories']),
        'category_counts': defaultdict(int),
        'category_names': {},
        'bbox_areas': [],
        'bbox_widths': [],
        'bbox_heights': [],
        'bbox_aspect_ratios': [],
        'objects_per_image': defaultdict(int),
        'image_sizes': [],
        'category_areas': defaultdict(list),
    }
    
    # 建立類別映射
    for cat in coco_data['categories']:
        stats['category_names'][cat['id']] = cat['name']
    
    # 分析標註
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        bbox = ann['bbox']  # [x, y, width, height]
        
        # 類別計數
        stats['category_counts'][cat_id] += 1
        
        # 邊界框統計
        width, height = bbox[2], bbox[3]
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        stats['bbox_areas'].append(area)
        stats['bbox_widths'].append(width)
        stats['bbox_heights'].append(height)
        stats['bbox_aspect_ratios'].append(aspect_ratio)
        stats['category_areas'][cat_id].append(area)
        
        # 每張圖片的物體數量
        stats['objects_per_image'][ann['image_id']] += 1
    
    # 分析圖片
    for img in coco_data['images']:
        stats['image_sizes'].append((img['width'], img['height']))
    
    return stats

def plot_category_distribution(stats, output_dir):
    """繪製類別分佈圖"""
    categories = []
    counts = []
    
    for cat_id in sorted(stats['category_counts'].keys()):
        categories.append(stats['category_names'][cat_id])
        counts.append(stats['category_counts'][cat_id])
    
    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # 長條圖
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    bars = ax1.barh(categories, counts, color=colors)
    ax1.set_xlabel('標註數量', fontsize=12)
    ax1.set_title('各類別標註數量分佈', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 在長條上顯示數值
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax1.text(count, i, f' {count}', va='center', fontsize=10)
    
    # 圓餅圖
    ax2.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.set_title('類別比例分佈', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'category_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已生成: {output_path}")
    plt.close()

def plot_bbox_statistics(stats, output_dir):
    """繪製邊界框統計圖"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 邊界框面積分佈
    ax = axes[0, 0]
    ax.hist(stats['bbox_areas'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('面積 (像素²)', fontsize=11)
    ax.set_ylabel('數量', fontsize=11)
    ax.set_title('邊界框面積分佈', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 顯示統計值
    mean_area = np.mean(stats['bbox_areas'])
    median_area = np.median(stats['bbox_areas'])
    ax.axvline(mean_area, color='red', linestyle='--', linewidth=2, label=f'平均值: {mean_area:.0f}')
    ax.axvline(median_area, color='green', linestyle='--', linewidth=2, label=f'中位數: {median_area:.0f}')
    ax.legend()
    
    # 2. 寬度和高度分佈
    ax = axes[0, 1]
    ax.hist(stats['bbox_widths'], bins=40, alpha=0.6, label='寬度', color='coral')
    ax.hist(stats['bbox_heights'], bins=40, alpha=0.6, label='高度', color='lightgreen')
    ax.set_xlabel('像素', fontsize=11)
    ax.set_ylabel('數量', fontsize=11)
    ax.set_title('邊界框寬度與高度分佈', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. 長寬比分佈
    ax = axes[1, 0]
    ax.hist(stats['bbox_aspect_ratios'], bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax.set_xlabel('長寬比 (寬度/高度)', fontsize=11)
    ax.set_ylabel('數量', fontsize=11)
    ax.set_title('邊界框長寬比分佈', fontsize=13, fontweight='bold')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='正方形')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. 寬度 vs 高度散點圖
    ax = axes[1, 1]
    ax.scatter(stats['bbox_widths'], stats['bbox_heights'], alpha=0.3, s=10, color='teal')
    ax.set_xlabel('寬度 (像素)', fontsize=11)
    ax.set_ylabel('高度 (像素)', fontsize=11)
    ax.set_title('邊界框寬度 vs 高度', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 添加對角線
    max_val = max(max(stats['bbox_widths']), max(stats['bbox_heights']))
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=1, label='寬=高')
    ax.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'bbox_statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已生成: {output_path}")
    plt.close()

def plot_objects_per_image(stats, output_dir):
    """繪製每張圖片的物體數量分佈"""
    objects_counts = list(stats['objects_per_image'].values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 直方圖
    ax1.hist(objects_counts, bins=range(0, max(objects_counts)+2), 
             color='gold', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('每張圖片的物體數量', fontsize=12)
    ax1.set_ylabel('圖片數量', fontsize=12)
    ax1.set_title('每張圖片的物體數量分佈', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # 顯示統計值
    mean_objects = np.mean(objects_counts)
    median_objects = np.median(objects_counts)
    ax1.axvline(mean_objects, color='red', linestyle='--', linewidth=2, 
                label=f'平均值: {mean_objects:.1f}')
    ax1.axvline(median_objects, color='green', linestyle='--', linewidth=2, 
                label=f'中位數: {median_objects:.1f}')
    ax1.legend()
    
    # Box plot
    ax2.boxplot(objects_counts, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('每張圖片的物體數量', fontsize=12)
    ax2.set_title('物體數量統計 (箱型圖)', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'objects_per_image.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已生成: {output_path}")
    plt.close()

def plot_category_area_comparison(stats, output_dir):
    """繪製各類別的邊界框面積比較"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 準備數據
    categories = []
    area_data = []
    
    for cat_id in sorted(stats['category_areas'].keys()):
        categories.append(stats['category_names'][cat_id])
        area_data.append(stats['category_areas'][cat_id])
    
    # 創建箱型圖
    bp = ax.boxplot(area_data, labels=categories, patch_artist=True, 
                    showmeans=True, meanline=True)
    
    # 美化箱型圖
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('邊界框面積 (像素²)', fontsize=12)
    ax.set_xlabel('類別', fontsize=12)
    ax.set_title('各類別邊界框面積分佈比較', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'category_area_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已生成: {output_path}")
    plt.close()

def plot_image_size_distribution(stats, output_dir):
    """繪製圖片尺寸分佈"""
    widths = [size[0] for size in stats['image_sizes']]
    heights = [size[1] for size in stats['image_sizes']]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 散點圖
    ax = axes[0]
    ax.scatter(widths, heights, alpha=0.6, s=50, color='purple', edgecolor='black')
    ax.set_xlabel('寬度 (像素)', fontsize=12)
    ax.set_ylabel('高度 (像素)', fontsize=12)
    ax.set_title('圖片尺寸分佈', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 顯示統計資訊
    unique_sizes = list(set(stats['image_sizes']))
    info_text = f"圖片總數: {len(stats['image_sizes'])}\n"
    info_text += f"不同尺寸: {len(unique_sizes)}\n"
    if len(unique_sizes) <= 5:
        info_text += "尺寸:\n"
        for size in unique_sizes:
            count = stats['image_sizes'].count(size)
            info_text += f"  {size[0]}×{size[1]}: {count}張\n"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    # 直方圖
    ax = axes[1]
    ax.hist(widths, bins=20, alpha=0.6, label='寬度', color='skyblue')
    ax.hist(heights, bins=20, alpha=0.6, label='高度', color='lightcoral')
    ax.set_xlabel('像素', fontsize=12)
    ax.set_ylabel('圖片數量', fontsize=12)
    ax.set_title('圖片寬度與高度分佈', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'image_size_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已生成: {output_path}")
    plt.close()

def generate_summary_report(stats, output_dir):
    """生成統計摘要報告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("GroceryDataset 標註統計報告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 基本統計
    report_lines.append("📊 基本統計")
    report_lines.append("-" * 80)
    report_lines.append(f"總圖片數量: {stats['total_images']}")
    report_lines.append(f"總標註數量: {stats['total_annotations']}")
    report_lines.append(f"類別總數: {stats['total_categories']}")
    report_lines.append(f"平均每張圖片的標註數: {stats['total_annotations'] / stats['total_images']:.2f}")
    report_lines.append("")
    
    # 類別統計
    report_lines.append("🏷️  類別統計")
    report_lines.append("-" * 80)
    for cat_id in sorted(stats['category_counts'].keys()):
        cat_name = stats['category_names'][cat_id]
        count = stats['category_counts'][cat_id]
        percentage = (count / stats['total_annotations']) * 100
        report_lines.append(f"  {cat_name:30} (ID:{cat_id:3}): {count:5} ({percentage:5.2f}%)")
    report_lines.append("")
    
    # 邊界框統計
    report_lines.append("📦 邊界框統計")
    report_lines.append("-" * 80)
    report_lines.append(f"面積 - 平均: {np.mean(stats['bbox_areas']):,.0f} 像素²")
    report_lines.append(f"面積 - 中位數: {np.median(stats['bbox_areas']):,.0f} 像素²")
    report_lines.append(f"面積 - 最小: {np.min(stats['bbox_areas']):,.0f} 像素²")
    report_lines.append(f"面積 - 最大: {np.max(stats['bbox_areas']):,.0f} 像素²")
    report_lines.append(f"面積 - 標準差: {np.std(stats['bbox_areas']):,.0f} 像素²")
    report_lines.append("")
    report_lines.append(f"寬度 - 平均: {np.mean(stats['bbox_widths']):.1f} 像素")
    report_lines.append(f"高度 - 平均: {np.mean(stats['bbox_heights']):.1f} 像素")
    report_lines.append(f"長寬比 - 平均: {np.mean(stats['bbox_aspect_ratios']):.2f}")
    report_lines.append("")
    
    # 每張圖片的物體數量統計
    objects_counts = list(stats['objects_per_image'].values())
    report_lines.append("🖼️  每張圖片的物體數量統計")
    report_lines.append("-" * 80)
    report_lines.append(f"平均: {np.mean(objects_counts):.2f}")
    report_lines.append(f"中位數: {np.median(objects_counts):.0f}")
    report_lines.append(f"最小: {np.min(objects_counts)}")
    report_lines.append(f"最大: {np.max(objects_counts)}")
    report_lines.append(f"標準差: {np.std(objects_counts):.2f}")
    report_lines.append("")
    
    # 圖片尺寸統計
    report_lines.append("📐 圖片尺寸統計")
    report_lines.append("-" * 80)
    unique_sizes = list(set(stats['image_sizes']))
    report_lines.append(f"不同尺寸數量: {len(unique_sizes)}")
    for size in unique_sizes:
        count = stats['image_sizes'].count(size)
        percentage = (count / stats['total_images']) * 100
        report_lines.append(f"  {size[0]:5}×{size[1]:5}: {count:4}張 ({percentage:5.2f}%)")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # 儲存報告
    report_text = "\n".join(report_lines)
    output_path = os.path.join(output_dir, 'statistics_report.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ 已生成: {output_path}")
    
    # 也在終端顯示
    print("\n" + report_text)

def create_comprehensive_visualization(coco_file, output_dir='statistics_visualizations'):
    """創建完整的統計視覺化"""
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在載入標註檔案: {coco_file}")
    coco_data = load_coco_data(coco_file)
    
    print("正在計算統計資訊...")
    stats = calculate_statistics(coco_data)
    
    print("\n正在生成視覺化圖表...")
    print("-" * 80)
    
    # 生成各種圖表
    plot_category_distribution(stats, output_dir)
    plot_bbox_statistics(stats, output_dir)
    plot_objects_per_image(stats, output_dir)
    plot_category_area_comparison(stats, output_dir)
    plot_image_size_distribution(stats, output_dir)
    
    # 生成統計報告
    print("\n正在生成統計報告...")
    print("-" * 80)
    generate_summary_report(stats, output_dir)
    
    print("\n" + "=" * 80)
    print(f"✅ 所有統計視覺化已完成！")
    print(f"📁 輸出目錄: {os.path.abspath(output_dir)}")
    print("=" * 80)

if __name__ == "__main__":
    # 設定檔案路徑
    coco_file = "annotations_coco.json"
    output_dir = "statistics_visualizations"
    
    # 生成完整的統計視覺化
    create_comprehensive_visualization(coco_file, output_dir)
