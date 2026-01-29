"""
Generate text embeddings for LocountOWOD tasks using the UniOW text encoder.
Allows regenerating specific task embeddings after class list changes.
"""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner

def extract_locount_feats(dataset_name: str = "LocountOWOD", tasks=None) -> bool:
    """Generate text embeddings for the specified LocountOWOD tasks."""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    config_file = project_root / "configs/pretrain/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.py"
    ckpt_file = project_root / "pretrained/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth"
    save_path = project_root / "embeddings/uniow-s"
    save_path.mkdir(parents=True, exist_ok=True)

    tasks = tasks or [1, 2, 3, 4]

    print("=" * 80)
    print(f"Generating embeddings for {dataset_name} tasks: {tasks}")
    print("=" * 80)

    try:
        cfg = Config.fromfile(str(config_file))
        cfg.work_dir = str(project_root / "work_dirs/extract_feats")

        runner = Runner.from_cfg(cfg)
        runner.call_hook("before_run")
        runner.load_checkpoint(str(ckpt_file), map_location="cpu")

        model = runner.model
        if torch.cuda.is_available():
            model = model.cuda()
            print("Using GPU")
        else:
            print("GPU not available, using CPU")
        model.eval()

        for task_id in tasks:
            print(f"\nTask {task_id}:")
            class_text_path = project_root / f"data/OWOD/ImageSets/{dataset_name}/t{task_id}_known.txt"
            if not class_text_path.exists():
                print(f"  Missing class list: {class_text_path}")
                return False

            with open(class_text_path, "r", encoding="utf-8") as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]

            print(f"  Known classes: {len(class_names)}")

            with torch.no_grad():
                text_feats = model.backbone.forward_text([class_names]).squeeze(0).detach().cpu()

            save_file = save_path / f"{dataset_name.lower()}_t{task_id}.npy"
            np.save(str(save_file), text_feats.numpy())
            print(f"  Saved: {save_file} | shape: {tuple(text_feats.shape)}")

        print("\nAll requested embeddings generated successfully.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def parse_args():
    parser = argparse.ArgumentParser(description="Extract LocountOWOD text embeddings")
    parser.add_argument("--dataset", default="LocountOWOD", help="Dataset name (default: LocountOWOD)")
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task ids to generate (default: all 1-4)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    tasks = None
    if args.tasks:
        tasks = [int(t.strip()) for t in args.tasks.split(",") if t.strip()]
    success = extract_locount_feats(dataset_name=args.dataset, tasks=tasks)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
