#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OWOD Log Analysis Tool
功能：
1. 读取所有OWOD训练log
2. 提取每个epoch的验证指标
3. 生成可视化图表
4. 按log日期和任务组织输出文件
"""

import re
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class OWODLogAnalyzer:
    def __init__(self, work_dir: str = 'work_dirs/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod'):
        self.work_dir = Path(work_dir)
        self.results = {}
        
    def find_log_files(self) -> Dict[str, Path]:
        """查找所有log文件"""
        log_files = {}
        for date_dir in self.work_dir.glob('2026*/'):
            log_path = list(date_dir.glob('*.log'))
            if log_path:
                log_files[date_dir.name] = log_path[0]
        return log_files
    
    def extract_task_from_log(self, log_path: Path) -> int:
        """从log文件推断Task编号"""
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 查找OWOD from形式的字符串
            match = re.search(r"Loading SOWODB from \['(.+?)'\]", content)
            if match:
                task_str = match.group(1)
                # t2_train -> Task 2
                if 't' in task_str:
                    task_match = re.search(r't(\d+)', task_str)
                    if task_match:
                        return int(task_match.group(1))
        # 默认为Task 1
        return 1
    
    def parse_validation_metrics(self, log_path: Path) -> Dict[int, Dict]:
        """解析log中的验证指标"""
        metrics_by_epoch = {}
        
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找所有验证输出块
        # 模式1：最详细的输出（包含Unknown AP50等）
        pattern1 = r'Saving checkpoint at (\d+) epochs\n.*?(?=Saving checkpoint at|\Z)'
        
        blocks = re.finditer(pattern1, content, re.DOTALL)
        
        for block in blocks:
            block_text = block.group(0)
            
            # 提取epoch数
            epoch_match = re.search(r'Saving checkpoint at (\d+) epochs', block_text)
            if not epoch_match:
                continue
            epoch = int(epoch_match.group(1))
            
            metrics = {}
            
            # 提取各类指标
            patterns = {
                'known_ap50': r'Known AP50: ([\d.]+)',
                'known_precision': r'Known Precisions50: ([\d.]+)',
                'known_recall': r'Known Recall50: ([\d.]+)',
                'unknown_ap50': r'Unknown AP50: ([\d.]+)',
                'unknown_precision': r'Unknown Precisions50: ([\d.]+)',
                'unknown_recall': r'Unknown Recall50: ([\d.]+)',
                'wilderness_impact': r"Wilderness Impact: \{50: ([\d.]+)\}",
                'absolute_ose': r"Absolute OSE \(total_num_unk_det_as_known\): \{50: ([\d.]+)\}",
                'total_unknown': r'total_num_unk (\d+)',
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, block_text)
                if match:
                    metrics[key] = float(match.group(1))
            
            # 计算综合指标 (Both = Known AP50，因为通常这是主要关注点)
            if 'known_ap50' in metrics:
                metrics['both'] = metrics['known_ap50']
            
            if metrics:
                metrics_by_epoch[epoch] = metrics
        
        return metrics_by_epoch
    
    def analyze_all_logs(self):
        """分析所有log文件"""
        log_files = self.find_log_files()
        
        for date_dir, log_path in sorted(log_files.items()):
            print(f"分析: {date_dir} -> {log_path.name}")
            
            task = self.extract_task_from_log(log_path)
            metrics = self.parse_validation_metrics(log_path)
            
            if metrics:
                self.results[date_dir] = {
                    'task': task,
                    'log_path': log_path,
                    'metrics': metrics,
                    'log_date': date_dir[:8],  # YYYYMMDD
                    'log_time': date_dir[8:],  # HHMMSS
                }
                print(f"  ✓ Task {task}: 找到 {len(metrics)} 个验证点")
        
        return self.results
    
    def plot_metrics(self, log_dir_name: str, result: Dict):
        """绘制单个log的验证指标图表"""
        metrics = result['metrics']
        task = result['task']
        
        if not metrics:
            print(f"  ⚠ {log_dir_name}: 无验证指标")
            return
        
        # 创建输出目录
        output_dir = self.work_dir / f'analysis_{log_dir_name}' / f'task_{task}'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 按epoch排序
        epochs = sorted(metrics.keys())
        
        # 准备数据
        known_ap50 = [metrics[e].get('known_ap50', np.nan) for e in epochs]
        known_recall = [metrics[e].get('known_recall', np.nan) for e in epochs]
        known_precision = [metrics[e].get('known_precision', np.nan) for e in epochs]
        
        unknown_ap50 = [metrics[e].get('unknown_ap50', np.nan) for e in epochs]
        unknown_recall = [metrics[e].get('unknown_recall', np.nan) for e in epochs]
        unknown_precision = [metrics[e].get('unknown_precision', np.nan) for e in epochs]
        
        wilderness_impact = [metrics[e].get('wilderness_impact', np.nan) for e in epochs]
        
        # 绘制1：已知/未知类对比
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Task {task} - OWOD 验证指标\n{log_dir_name}', fontsize=14, fontweight='bold')
        
        # AP50对比
        ax = axes[0, 0]
        ax.plot(epochs, known_ap50, 'o-', label='Known AP50', linewidth=2, markersize=6)
        ax.plot(epochs, unknown_ap50, 's-', label='Unknown AP50', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('AP50 (%)', fontsize=11)
        ax.set_title('AP50 Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Recall对比
        ax = axes[0, 1]
        ax.plot(epochs, known_recall, 'o-', label='Known Recall', linewidth=2, markersize=6)
        ax.plot(epochs, unknown_recall, 's-', label='Unknown Recall', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Recall (%)', fontsize=11)
        ax.set_title('Recall Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Precision对比
        ax = axes[1, 0]
        ax.plot(epochs, known_precision, 'o-', label='Known Precision', linewidth=2, markersize=6)
        ax.plot(epochs, unknown_precision, 's-', label='Unknown Precision', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Precision (%)', fontsize=11)
        ax.set_title('Precision Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Wilderness Impact (越低越好)
        ax = axes[1, 1]
        ax.plot(epochs, wilderness_impact, 'D-', label='Wilderness Impact', linewidth=2, markersize=6, color='red')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('WI (越低越好)', fontsize=11)
        ax.set_title('Wilderness Impact Trend', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = output_dir / f'owod_metrics_{log_dir_name}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ 图表已保存: {plot_file.relative_to(self.work_dir)}")
        plt.close()
        
        # 生成详细JSON报告
        self._save_json_report(output_dir, log_dir_name, task, epochs, metrics)
    
    def _save_json_report(self, output_dir: Path, log_dir_name: str, task: int, 
                         epochs: List[int], metrics: Dict):
        """保存JSON格式的详细报告"""
        report = {
            'log_info': {
                'log_directory': log_dir_name,
                'task': task,
                'total_epochs': len(epochs),
                'analysis_time': datetime.now().isoformat(),
            },
            'validation_results': {}
        }
        
        for epoch in epochs:
            report['validation_results'][f'epoch_{epoch}'] = metrics[epoch]
        
        # 找出最佳模型
        if metrics:
            best_epoch = max(epochs, key=lambda e: metrics[e].get('both', -float('inf')))
            report['best_model'] = {
                'epoch': best_epoch,
                'score': metrics[best_epoch].get('both', 0),
                'checkpoint': f'epoch_{best_epoch}.pth'
            }
        
        # 保存JSON
        json_file = output_dir / f'validation_report_{log_dir_name}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ 报告已保存: {json_file.relative_to(self.work_dir)}")
    
    def generate_summary_report(self):
        """生成所有Task的汇总报告"""
        if not self.results:
            print("没有找到任何log文件！")
            return
        
        summary_file = self.work_dir / 'OWOD_Summary_Report.txt'
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("OWOD Training Summary Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # 按Task分组
            tasks_data = {}
            for log_dir, result in sorted(self.results.items()):
                task = result['task']
                if task not in tasks_data:
                    tasks_data[task] = []
                tasks_data[task].append((log_dir, result))
            
            # 写入每个Task的信息
            for task in sorted(tasks_data.keys()):
                f.write(f"\nTASK {task}\n")
                f.write("-" * 80 + "\n")
                
                for log_dir, result in tasks_data[task]:
                    f.write(f"\nLog Directory: {log_dir}\n")
                    f.write(f"Date: {result['log_date']}, Time: {result['log_time']}\n")
                    
                    metrics = result['metrics']
                    if metrics:
                        best_epoch = max(metrics.keys(), 
                                       key=lambda e: metrics[e].get('both', -float('inf')))
                        best_metrics = metrics[best_epoch]
                        
                        f.write(f"\nBest Epoch: {best_epoch}\n")
                        f.write(f"  Known AP50:        {best_metrics.get('known_ap50', 'N/A'):.4f}\n")
                        f.write(f"  Known Recall:      {best_metrics.get('known_recall', 'N/A'):.4f}%\n")
                        f.write(f"  Known Precision:   {best_metrics.get('known_precision', 'N/A'):.4f}%\n")
                        
                        if 'unknown_ap50' in best_metrics:
                            f.write(f"  Unknown AP50:      {best_metrics['unknown_ap50']:.4f}\n")
                            f.write(f"  Unknown Recall:    {best_metrics['unknown_recall']:.4f}%\n")
                            f.write(f"  Unknown Precision: {best_metrics['unknown_precision']:.4f}%\n")
                            f.write(f"  Wilderness Impact: {best_metrics.get('wilderness_impact', 'N/A'):.6f}\n")
                        
                        f.write(f"\nCheckpoint: epoch_{best_epoch}.pth\n")
        
        print(f"\n✓ 汇总报告已保存: {summary_file.relative_to(self.work_dir)}")
    
    def run(self):
        """执行完整的分析流程"""
        print("\n" + "="*80)
        print("OWOD Log Analysis Tool")
        print("="*80 + "\n")
        
        # 1. 分析所有log
        print("步骤 1/3: 分析所有log文件...")
        self.analyze_all_logs()
        
        if not self.results:
            print("❌ 未找到任何log文件！")
            return
        
        print(f"✓ 找到 {len(self.results)} 个log文件\n")
        
        # 2. 生成可视化
        print("步骤 2/3: 生成可视化图表...")
        for log_dir, result in sorted(self.results.items()):
            self.plot_metrics(log_dir, result)
        
        # 3. 生成汇总报告
        print("\n步骤 3/3: 生成汇总报告...")
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print("✓ 分析完成！")
        print("="*80 + "\n")


if __name__ == '__main__':
    analyzer = OWODLogAnalyzer()
    analyzer.run()
