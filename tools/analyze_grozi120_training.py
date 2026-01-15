#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GroZi-120 OWOD 訓練日誌分析工具
功能：
1. 讀取訓練日誌並提取 loss 和指標
2. 繪製訓練過程中的 loss 變化曲線
3. 繪製驗證指標變化曲線
4. 生成詳細的訓練報告
"""

import re
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple
import argparse

# 設定中文字體
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class GroZi120LogAnalyzer:
    def __init__(self, work_dir: str = 'work_dirs/grozi120_task1'):
        self.work_dir = Path(work_dir)
        self.train_metrics = []
        self.val_metrics = {}
        
    def find_latest_log(self) -> Path:
        """找到最新的訓練日誌"""
        log_files = list(self.work_dir.glob('*/*.log'))
        if not log_files:
            log_files = list(self.work_dir.glob('*.log'))
        
        if not log_files:
            raise FileNotFoundError(f"在 {self.work_dir} 中找不到日誌文件")
        
        # 返回最新的日誌文件
        return max(log_files, key=lambda p: p.stat().st_mtime)
    
    def parse_training_log(self, log_path: Path):
        """解析訓練日誌中的 loss 和指標"""
        print(f"📖 讀取日誌: {log_path}")
        
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 解析訓練指標
        train_pattern = re.compile(
            r'Epoch\(train\)\s+\[(\d+)\]\[(\d+)/(\d+)\].*?'
            r'loss:\s+([\d.]+).*?'
            r'one2many_loss_cls:\s+([\d.]+).*?'
            r'one2many_loss_bbox:\s+([\d.]+).*?'
            r'one2many_loss_dfl:\s+([\d.]+).*?'
            r'one2one_loss_cls:\s+([\d.]+).*?'
            r'one2one_loss_bbox:\s+([\d.]+).*?'
            r'one2one_loss_dfl:\s+([\d.]+)'
        )
        
        for line in lines:
            match = train_pattern.search(line)
            if match:
                epoch = int(match.group(1))
                iter_cur = int(match.group(2))
                iter_total = int(match.group(3))
                
                metrics = {
                    'epoch': epoch,
                    'iter': iter_cur,
                    'iter_total': iter_total,
                    'total_loss': float(match.group(4)),
                    'one2many_loss_cls': float(match.group(5)),
                    'one2many_loss_bbox': float(match.group(6)),
                    'one2many_loss_dfl': float(match.group(7)),
                    'one2one_loss_cls': float(match.group(8)),
                    'one2one_loss_bbox': float(match.group(9)),
                    'one2one_loss_dfl': float(match.group(10)),
                }
                self.train_metrics.append(metrics)
        
        print(f"  ✓ 找到 {len(self.train_metrics)} 個訓練迭代記錄")
        
        # 解析驗證指標
        self._parse_validation_metrics(lines)
    
    def _parse_validation_metrics(self, lines: List[str]):
        """解析驗證指標"""
        current_epoch = None
        in_val_section = False
        
        for i, line in enumerate(lines):
            # 檢測驗證開始
            if 'Saving checkpoint at' in line:
                match = re.search(r'Saving checkpoint at (\d+) epochs', line)
                if match:
                    current_epoch = int(match.group(1))
                    in_val_section = True
                    self.val_metrics[current_epoch] = {}
            
            # 提取驗證指標
            if in_val_section and current_epoch:
                # Known 類別指標
                if 'Known AP50:' in line:
                    match = re.search(r'Known AP50:\s+([\d.]+)', line)
                    if match:
                        self.val_metrics[current_epoch]['known_ap50'] = float(match.group(1))
                
                if 'Known Recall50:' in line:
                    match = re.search(r'Known Recall50:\s+([\d.]+)', line)
                    if match:
                        self.val_metrics[current_epoch]['known_recall'] = float(match.group(1))
                
                if 'Known Precisions50:' in line:
                    match = re.search(r'Known Precisions50:\s+([\d.]+)', line)
                    if match:
                        self.val_metrics[current_epoch]['known_precision'] = float(match.group(1))
                
                # Unknown 類別指標
                if 'Unknown AP50:' in line:
                    match = re.search(r'Unknown AP50:\s+([\d.]+)', line)
                    if match:
                        self.val_metrics[current_epoch]['unknown_ap50'] = float(match.group(1))
                
                if 'Unknown Recall50:' in line:
                    match = re.search(r'Unknown Recall50:\s+([\d.]+)', line)
                    if match:
                        self.val_metrics[current_epoch]['unknown_recall'] = float(match.group(1))
                
                if 'Wilderness Impact:' in line:
                    match = re.search(r'Wilderness Impact:\s+\{50:\s+([\d.]+)\}', line)
                    if match:
                        self.val_metrics[current_epoch]['wilderness_impact'] = float(match.group(1))
                        in_val_section = False  # 驗證section結束
        
        print(f"  ✓ 找到 {len(self.val_metrics)} 個驗證點")
    
    def plot_training_loss(self, output_dir: Path):
        """繪製訓練 loss 曲線"""
        if not self.train_metrics:
            print("  ⚠️  沒有訓練數據可繪製")
            return
        
        # 準備數據 - 計算每個 epoch 的平均值
        epochs_data = {}
        for metric in self.train_metrics:
            epoch = metric['epoch']
            if epoch not in epochs_data:
                epochs_data[epoch] = {
                    'total_loss': [],
                    'one2many_loss_cls': [],
                    'one2many_loss_bbox': [],
                    'one2many_loss_dfl': [],
                    'one2one_loss_cls': [],
                    'one2one_loss_bbox': [],
                    'one2one_loss_dfl': [],
                }
            
            epochs_data[epoch]['total_loss'].append(metric['total_loss'])
            epochs_data[epoch]['one2many_loss_cls'].append(metric['one2many_loss_cls'])
            epochs_data[epoch]['one2many_loss_bbox'].append(metric['one2many_loss_bbox'])
            epochs_data[epoch]['one2many_loss_dfl'].append(metric['one2many_loss_dfl'])
            epochs_data[epoch]['one2one_loss_cls'].append(metric['one2one_loss_cls'])
            epochs_data[epoch]['one2one_loss_bbox'].append(metric['one2one_loss_bbox'])
            epochs_data[epoch]['one2one_loss_dfl'].append(metric['one2one_loss_dfl'])
        
        # 計算每個 epoch 的平均值
        epochs = sorted(epochs_data.keys())
        avg_total_loss = [np.mean(epochs_data[e]['total_loss']) for e in epochs]
        avg_cls_loss = [np.mean(epochs_data[e]['one2many_loss_cls']) + 
                       np.mean(epochs_data[e]['one2one_loss_cls']) for e in epochs]
        avg_bbox_loss = [np.mean(epochs_data[e]['one2many_loss_bbox']) + 
                        np.mean(epochs_data[e]['one2one_loss_bbox']) for e in epochs]
        avg_dfl_loss = [np.mean(epochs_data[e]['one2many_loss_dfl']) + 
                       np.mean(epochs_data[e]['one2one_loss_dfl']) for e in epochs]
        
        # 繪製 loss 曲線
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GroZi-120 OWOD Training Loss', fontsize=16, fontweight='bold')
        
        # 總 Loss
        ax = axes[0, 0]
        ax.plot(epochs, avg_total_loss, 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Total Loss', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 分類 Loss
        ax = axes[0, 1]
        ax.plot(epochs, avg_cls_loss, 'r-', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Classification Loss (One2Many + One2One)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # BBox Loss
        ax = axes[1, 0]
        ax.plot(epochs, avg_bbox_loss, 'g-', linewidth=2, marker='^', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('BBox Loss (One2Many + One2One)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # DFL Loss
        ax = axes[1, 1]
        ax.plot(epochs, avg_dfl_loss, 'm-', linewidth=2, marker='d', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('DFL Loss (One2Many + One2One)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存圖表
        output_file = output_dir / 'training_loss_curves.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Loss 曲線已保存: {output_file}")
        plt.close()
    
    def plot_validation_metrics(self, output_dir: Path):
        """繪製驗證指標曲線"""
        if not self.val_metrics:
            print("  ⚠️  沒有驗證數據可繪製")
            return
        
        epochs = sorted(self.val_metrics.keys())
        
        # 準備數據
        known_ap50 = [self.val_metrics[e].get('known_ap50', np.nan) for e in epochs]
        known_recall = [self.val_metrics[e].get('known_recall', np.nan) for e in epochs]
        known_precision = [self.val_metrics[e].get('known_precision', np.nan) for e in epochs]
        unknown_ap50 = [self.val_metrics[e].get('unknown_ap50', np.nan) for e in epochs]
        unknown_recall = [self.val_metrics[e].get('unknown_recall', np.nan) for e in epochs]
        wilderness = [self.val_metrics[e].get('wilderness_impact', np.nan) for e in epochs]
        
        # 繪製驗證指標
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GroZi-120 OWOD Validation Metrics', fontsize=16, fontweight='bold')
        
        # Known vs Unknown AP50
        ax = axes[0, 0]
        ax.plot(epochs, known_ap50, 'b-o', linewidth=2, markersize=6, label='Known AP50')
        ax.plot(epochs, unknown_ap50, 'r-s', linewidth=2, markersize=6, label='Unknown AP50')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('AP50', fontsize=12)
        ax.set_title('AP50 (Known vs Unknown)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Known Metrics
        ax = axes[0, 1]
        ax.plot(epochs, known_ap50, 'b-o', linewidth=2, markersize=5, label='AP50')
        ax.plot(epochs, known_recall, 'g-^', linewidth=2, markersize=5, label='Recall')
        ax.plot(epochs, known_precision, 'm-s', linewidth=2, markersize=5, label='Precision')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Known Classes Metrics', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Unknown Recall
        ax = axes[1, 0]
        ax.plot(epochs, unknown_recall, 'r-d', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Recall (%)', fontsize=12)
        ax.set_title('Unknown Recall', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Wilderness Impact
        ax = axes[1, 1]
        ax.plot(epochs, wilderness, 'orange', linewidth=2, marker='D', markersize=6)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Wilderness Impact (越低越好)', fontsize=12)
        ax.set_title('Wilderness Impact', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存圖表
        output_file = output_dir / 'validation_metrics_curves.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ 驗證指標曲線已保存: {output_file}")
        plt.close()
    
    def generate_summary_report(self, output_dir: Path, log_path: Path):
        """生成訓練摘要報告"""
        report = {
            'log_file': str(log_path),
            'analysis_time': datetime.now().isoformat(),
            'training_summary': {
                'total_iterations': len(self.train_metrics),
                'epochs_trained': max([m['epoch'] for m in self.train_metrics]) if self.train_metrics else 0,
            },
            'validation_summary': {
                'total_validations': len(self.val_metrics),
            }
        }
        
        # 找出最佳模型
        if self.val_metrics:
            best_epoch = max(self.val_metrics.keys(), 
                           key=lambda e: self.val_metrics[e].get('known_ap50', 0))
            report['best_model'] = {
                'epoch': best_epoch,
                'metrics': self.val_metrics[best_epoch]
            }
        
        # 最後一個 epoch 的數據
        if self.train_metrics:
            last_metrics = self.train_metrics[-1]
            report['latest_training'] = {
                'epoch': last_metrics['epoch'],
                'total_loss': last_metrics['total_loss'],
            }
        
        # 保存 JSON
        json_file = output_dir / 'training_report.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  ✓ JSON 報告已保存: {json_file}")
        
        # 保存文本報告
        txt_file = output_dir / 'training_report.txt'
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("GroZi-120 OWOD Training Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Log File: {log_path}\n")
            f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Training Summary:\n")
            f.write(f"  Total Iterations: {report['training_summary']['total_iterations']}\n")
            f.write(f"  Epochs Trained: {report['training_summary']['epochs_trained']}\n\n")
            
            if 'best_model' in report:
                f.write("Best Model:\n")
                f.write(f"  Epoch: {report['best_model']['epoch']}\n")
                for key, value in report['best_model']['metrics'].items():
                    f.write(f"  {key}: {value:.4f}\n")
        
        print(f"  ✓ 文本報告已保存: {txt_file}")
    
    def run(self):
        """執行完整分析"""
        print("\n" + "=" * 80)
        print("GroZi-120 OWOD 訓練日誌分析工具")
        print("=" * 80 + "\n")
        
        # 1. 找到日誌文件
        try:
            log_path = self.find_latest_log()
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return
        
        # 2. 解析日誌
        print("步驟 1/4: 解析訓練日誌...")
        self.parse_training_log(log_path)
        
        # 3. 創建輸出目錄
        output_dir = self.work_dir / 'analysis'
        output_dir.mkdir(exist_ok=True)
        
        # 4. 繪製圖表
        print("\n步驟 2/4: 繪製訓練 Loss 曲線...")
        self.plot_training_loss(output_dir)
        
        print("\n步驟 3/4: 繪製驗證指標曲線...")
        self.plot_validation_metrics(output_dir)
        
        # 5. 生成報告
        print("\n步驟 4/4: 生成訓練報告...")
        self.generate_summary_report(output_dir, log_path)
        
        print("\n" + "=" * 80)
        print(f"✅ 分析完成！結果保存在: {output_dir}")
        print("=" * 80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GroZi-120 OWOD 訓練日誌分析')
    parser.add_argument('--work-dir', type=str, default='work_dirs/grozi120_task1',
                       help='訓練工作目錄路徑')
    
    args = parser.parse_args()
    
    analyzer = GroZi120LogAnalyzer(work_dir=args.work_dir)
    analyzer.run()
