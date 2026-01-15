#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用訓練 Log 分析工具
功能：
1. 自動搜索所有訓練 log 文件
2. 提取訓練 loss 和驗證指標
3. 繪製訓練曲線圖
4. 支持多個 log 文件對比
"""

import re
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple, Optional
import json

# 設定中文字體
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')


class TrainingLogAnalyzer:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.training_data = []  # 訓練 loss 數據
        self.validation_data = {}  # 驗證指標數據
        
    def parse_training_losses(self) -> List[Dict]:
        """提取訓練 loss"""
        training_data = []
        
        with open(self.log_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 匹配訓練 loss 行
                # 例如: Epoch(train)  [1][41/41]  base_lr: 1.0000e-03 ... loss: 192.2991 ...
                if 'Epoch(train)' in line:
                    match = re.search(r'Epoch\(train\)\s+\[(\d+)\]\[(\d+)/(\d+)\]', line)
                    if not match:
                        continue
                    
                    epoch = int(match.group(1))
                    iter_num = int(match.group(2))
                    total_iters = int(match.group(3))
                    
                    # 提取各種 loss
                    loss_data = {'epoch': epoch, 'iter': iter_num, 'total_iters': total_iters}
                    
                    # 總 loss
                    loss_match = re.search(r'loss:\s+([\d.]+)', line)
                    if loss_match:
                        loss_data['total_loss'] = float(loss_match.group(1))
                    
                    # One2many losses
                    patterns = {
                        'one2many_loss_cls': r'one2many_loss_cls:\s+([\d.]+)',
                        'one2many_loss_bbox': r'one2many_loss_bbox:\s+([\d.]+)',
                        'one2many_loss_dfl': r'one2many_loss_dfl:\s+([\d.]+)',
                        'one2one_loss_cls': r'one2one_loss_cls:\s+([\d.]+)',
                        'one2one_loss_bbox': r'one2one_loss_bbox:\s+([\d.]+)',
                        'one2one_loss_dfl': r'one2one_loss_dfl:\s+([\d.]+)',
                    }
                    
                    for key, pattern in patterns.items():
                        match = re.search(pattern, line)
                        if match:
                            loss_data[key] = float(match.group(1))
                    
                    # Learning rate
                    lr_match = re.search(r'lr:\s+([\d.e-]+)', line)
                    if lr_match:
                        loss_data['learning_rate'] = float(lr_match.group(1))
                    
                    training_data.append(loss_data)
        
        return training_data
    
    def parse_validation_metrics(self) -> Dict[int, Dict]:
        """提取驗證指標"""
        metrics_by_epoch = {}
        
        with open(self.log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找驗證輸出區塊
        # 匹配模式：從 "Saving checkpoint at X epochs" 到下一個相同標記或文件結尾
        pattern = r'Saving checkpoint at (\d+) epochs.*?(?=Saving checkpoint at|\Z)'
        blocks = re.finditer(pattern, content, re.DOTALL)
        
        for block in blocks:
            block_text = block.group(0)
            
            # 提取 epoch
            epoch_match = re.search(r'Saving checkpoint at (\d+) epochs', block_text)
            if not epoch_match:
                continue
            epoch = int(epoch_match.group(1))
            
            metrics = {'epoch': epoch}
            
            # 提取各類指標
            patterns = {
                'known_ap50': r'Known AP50:\s+([\d.]+)',
                'known_precision': r'Known Precisions50:\s+([\d.]+)',
                'known_recall': r'Known Recall50:\s+([\d.]+)',
                'unknown_ap50': r'Unknown AP50:\s+([\d.]+)',
                'unknown_precision': r'Unknown Precisions50:\s+([\d.]+)',
                'unknown_recall': r'Unknown Recall50:\s+([\d.]+)',
                'wilderness_impact': r"Wilderness Impact:\s+\{50:\s+([\d.]+)\}",
                'absolute_ose': r"Absolute OSE.*?:\s+\{50:\s+([\d.]+)\}",
                'both': r'owod/Both:\s+([\d.]+)',
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, block_text)
                if match:
                    metrics[key] = float(match.group(1))
            
            if len(metrics) > 1:  # 確保有提取到指標
                metrics_by_epoch[epoch] = metrics
        
        return metrics_by_epoch
    
    def analyze(self):
        """執行完整分析"""
        print(f"分析 log 文件: {self.log_path.name}")
        
        self.training_data = self.parse_training_losses()
        self.validation_data = self.parse_validation_metrics()
        
        print(f"  [OK] 訓練數據點: {len(self.training_data)}")
        print(f"  [OK] 驗證點: {len(self.validation_data)}")
        
        return self
    
    def plot_training_curves(self, save_dir: Optional[Path] = None):
        """繪製訓練曲線"""
        if not self.training_data:
            print("  [!] 沒有訓練數據")
            return
        
        if save_dir is None:
            save_dir = self.log_path.parent / 'analysis'
        save_dir.mkdir(exist_ok=True)
        
        # 按 epoch 分組計算平均值
        epochs = sorted(set(d['epoch'] for d in self.training_data))
        
        epoch_losses = {}
        for epoch in epochs:
            epoch_data = [d for d in self.training_data if d['epoch'] == epoch]
            
            # 計算平均值
            avg_losses = {}
            for key in ['total_loss', 'one2many_loss_cls', 'one2many_loss_bbox', 
                       'one2many_loss_dfl', 'one2one_loss_cls', 'one2one_loss_bbox', 
                       'one2one_loss_dfl']:
                values = [d.get(key, np.nan) for d in epoch_data]
                avg_losses[key] = np.nanmean(values)
            
            epoch_losses[epoch] = avg_losses
        
        # 繪製圖表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'訓練曲線 - {self.log_path.stem}', fontsize=14, fontweight='bold')
        
        epochs_list = sorted(epoch_losses.keys())
        
        # 1. 總 Loss
        ax = axes[0, 0]
        total_loss = [epoch_losses[e]['total_loss'] for e in epochs_list]
        ax.plot(epochs_list, total_loss, 'o-', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Total Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2. Classification Losses
        ax = axes[0, 1]
        one2many_cls = [epoch_losses[e]['one2many_loss_cls'] for e in epochs_list]
        one2one_cls = [epoch_losses[e]['one2one_loss_cls'] for e in epochs_list]
        ax.plot(epochs_list, one2many_cls, 'o-', label='One2Many Cls', linewidth=2, markersize=4)
        ax.plot(epochs_list, one2one_cls, 's-', label='One2One Cls', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Classification Loss', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 3. BBox Losses
        ax = axes[1, 0]
        one2many_bbox = [epoch_losses[e]['one2many_loss_bbox'] for e in epochs_list]
        one2one_bbox = [epoch_losses[e]['one2one_loss_bbox'] for e in epochs_list]
        ax.plot(epochs_list, one2many_bbox, 'o-', label='One2Many BBox', linewidth=2, markersize=4)
        ax.plot(epochs_list, one2one_bbox, 's-', label='One2One BBox', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('BBox Loss', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 4. DFL Losses
        ax = axes[1, 1]
        one2many_dfl = [epoch_losses[e]['one2many_loss_dfl'] for e in epochs_list]
        one2one_dfl = [epoch_losses[e]['one2one_loss_dfl'] for e in epochs_list]
        ax.plot(epochs_list, one2many_dfl, 'o-', label='One2Many DFL', linewidth=2, markersize=4)
        ax.plot(epochs_list, one2one_dfl, 's-', label='One2One DFL', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('DFL Loss', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = save_dir / f'training_losses_{self.log_path.stem}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  [OK] 訓練曲線已保存: {plot_file}")
        plt.close()
    
    def plot_validation_curves(self, save_dir: Optional[Path] = None):
        """繪製驗證曲線"""
        if not self.validation_data:
            print("  [!] 沒有驗證數據")
            return
        
        if save_dir is None:
            save_dir = self.log_path.parent / 'analysis'
        save_dir.mkdir(exist_ok=True)
        
        epochs = sorted(self.validation_data.keys())
        
        # 準備數據
        known_ap50 = [self.validation_data[e].get('known_ap50', np.nan) for e in epochs]
        known_recall = [self.validation_data[e].get('known_recall', np.nan) for e in epochs]
        known_precision = [self.validation_data[e].get('known_precision', np.nan) for e in epochs]
        unknown_ap50 = [self.validation_data[e].get('unknown_ap50', np.nan) for e in epochs]
        unknown_recall = [self.validation_data[e].get('unknown_recall', np.nan) for e in epochs]
        wilderness_impact = [self.validation_data[e].get('wilderness_impact', np.nan) for e in epochs]
        
        # 繪製圖表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'驗證指標 - {self.log_path.stem}', fontsize=14, fontweight='bold')
        
        # 1. AP50
        ax = axes[0, 0]
        ax.plot(epochs, known_ap50, 'o-', label='Known AP50', linewidth=2, markersize=6)
        if not all(np.isnan(unknown_ap50)):
            ax.plot(epochs, unknown_ap50, 's-', label='Unknown AP50', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('AP50', fontsize=11)
        ax.set_title('AP50', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 2. Recall
        ax = axes[0, 1]
        ax.plot(epochs, known_recall, 'o-', label='Known Recall', linewidth=2, markersize=6)
        if not all(np.isnan(unknown_recall)):
            ax.plot(epochs, unknown_recall, 's-', label='Unknown Recall', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Recall (%)', fontsize=11)
        ax.set_title('Recall', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 3. Precision
        ax = axes[1, 0]
        ax.plot(epochs, known_precision, 'o-', label='Known Precision', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Precision (%)', fontsize=11)
        ax.set_title('Precision', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 4. Wilderness Impact
        ax = axes[1, 1]
        if not all(np.isnan(wilderness_impact)):
            ax.plot(epochs, wilderness_impact, 'D-', label='Wilderness Impact', 
                   linewidth=2, markersize=6, color='red')
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('WI (越低越好)', fontsize=11)
            ax.set_title('Wilderness Impact', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = save_dir / f'validation_metrics_{self.log_path.stem}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  [OK] 驗證曲線已保存: {plot_file}")
        plt.close()
    
    def save_summary(self, save_dir: Optional[Path] = None):
        """保存數據摘要"""
        if save_dir is None:
            save_dir = self.log_path.parent / 'analysis'
        save_dir.mkdir(exist_ok=True)
        
        summary = {
            'log_file': str(self.log_path),
            'analysis_time': datetime.now().isoformat(),
            'training_epochs': len(set(d['epoch'] for d in self.training_data)) if self.training_data else 0,
            'validation_epochs': list(sorted(self.validation_data.keys())),
        }
        
        # 最佳驗證結果
        if self.validation_data:
            best_epoch = max(self.validation_data.keys(), 
                           key=lambda e: self.validation_data[e].get('both', 
                                                                     self.validation_data[e].get('known_ap50', -1)))
            summary['best_validation'] = {
                'epoch': best_epoch,
                'metrics': self.validation_data[best_epoch]
            }
        
        json_file = save_dir / f'summary_{self.log_path.stem}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"  [OK] 摘要已保存: {json_file}")


def find_all_logs(base_dir: str = 'work_dirs') -> List[Path]:
    """自動搜索所有 log 文件"""
    base_path = Path(base_dir)
    log_files = []
    
    # 搜索所有 .log 文件
    for log_file in base_path.rglob('*.log'):
        # 排除一些不相關的 log
        if 'vis_data' not in str(log_file):
            log_files.append(log_file)
    
    return sorted(log_files)


def main():
    parser = argparse.ArgumentParser(description='訓練 Log 分析工具')
    parser.add_argument('--log', type=str, help='指定單個 log 文件路徑')
    parser.add_argument('--dir', type=str, default='work_dirs', help='搜索目錄（默認: work_dirs）')
    parser.add_argument('--output', type=str, help='輸出目錄（默認: log文件所在目錄/analysis）')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("訓練 Log 分析工具")
    print("="*80 + "\n")
    
    # 確定要分析的 log 文件
    if args.log:
        log_files = [Path(args.log)]
    else:
        print(f"搜索目錄: {args.dir}")
        log_files = find_all_logs(args.dir)
        print(f"找到 {len(log_files)} 個 log 文件\n")
    
    if not log_files:
        print("[ERROR] 沒有找到 log 文件！")
        return
    
    # 分析每個 log 文件
    for i, log_file in enumerate(log_files, 1):
        print(f"\n[{i}/{len(log_files)}] 處理: {log_file}")
        print("-" * 80)
        
        try:
            analyzer = TrainingLogAnalyzer(log_file)
            analyzer.analyze()
            
            output_dir = Path(args.output) if args.output else None
            
            analyzer.plot_training_curves(output_dir)
            analyzer.plot_validation_curves(output_dir)
            analyzer.save_summary(output_dir)
            
        except Exception as e:
            print(f"  [ERROR] 錯誤: {e}")
            continue
    
    print("\n" + "="*80)
    print("[OK] 分析完成！")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
