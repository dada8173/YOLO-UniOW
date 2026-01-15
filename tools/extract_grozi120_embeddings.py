"""
簡化版本的文本特徵提取腳本 - 為 GroZi-120 OWOD 生成所有任務的嵌入
"""
import sys
import os
from pathlib import Path
import numpy as np
import torch

# 添加當前目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mmengine.config import Config
from mmengine.runner import Runner

def extract_grozi120_feats():
    """為 GroZi-120 的所有 4 個任務生成文本嵌入"""
    
    # 確保在項目根目錄
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    config_file = project_root / 'configs/pretrain/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py'
    ckpt_file = project_root / 'pretrained/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth'
    save_path = project_root / 'embeddings/uniow-s'
    save_path.mkdir(parents=True, exist_ok=True)
    
    dataset_name = 'GroZi120OWOD'
    
    print("=" * 80)
    print(f"📊 生成 {dataset_name} 文本特徵嵌入")
    print("=" * 80)
    
    try:
        # 載入配置和模型
        print(f"\n1️⃣ 載入配置: {config_file.name}")
        cfg = Config.fromfile(str(config_file))
        cfg.work_dir = str(project_root / 'work_dirs/extract_feats')
        
        print(f"2️⃣ 初始化運行器...")
        runner = Runner.from_cfg(cfg)
        runner.call_hook("before_run")
        
        print(f"3️⃣ 載入檢查點: {ckpt_file.name}")
        runner.load_checkpoint(str(ckpt_file), map_location='cpu')
        
        print(f"4️⃣ 模型轉移到 GPU...")
        model = runner.model
        if torch.cuda.is_available():
            model = model.cuda()
            print("   ✅ 使用 GPU")
        else:
            print("   ⚠️  GPU 不可用，使用 CPU")
        model.eval()
        
        # 為每個任務提取特徵
        print(f"\n5️⃣ 提取文本特徵:")
        for task_id in range(1, 5):
            print(f"\n   Task {task_id}:")
            
            # 讀取該任務的已知類別
            class_text_path = project_root / f'data/OWOD/ImageSets/{dataset_name}/t{task_id}_known.txt'
            print(f"     讀取類別文件: {class_text_path.name}")
            
            with open(str(class_text_path), 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f.readlines()]
            
            print(f"     已知類別數: {len(class_names)}")
            
            # 提取特徵
            with torch.no_grad():
                text_feats = model.backbone.forward_text([class_names]).squeeze(0).detach().cpu()
            
            # 保存
            save_file = save_path / f'{dataset_name.lower()}_t{task_id}.npy'
            np.save(str(save_file), text_feats.numpy())
            print(f"     ✅ 保存到: {save_file}")
            print(f"     特徵形狀: {text_feats.shape}")
        
        print("\n" + "=" * 80)
        print("🎉 所有任務的文本特徵已生成！")
        print("=" * 80)
        print(f"\n生成的文件:")
        for task_id in range(1, 5):
            save_file = save_path / f'{dataset_name.lower()}_t{task_id}.npy'
            if save_file.exists():
                size = save_file.stat().st_size / (1024*1024)  # 轉換為 MB
                print(f"  ✅ {save_file.name:40s} ({size:.2f} MB)")
        
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    os.chdir(Path(__file__).parent.parent.parent)  # 切換到項目根目錄
    success = extract_grozi120_feats()
    sys.exit(0 if success else 1)
