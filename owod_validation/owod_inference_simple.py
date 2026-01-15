#!/usr/bin/env python
"""
OWOD 模型推断和可視化腳本
用法: python owod_inference_simple.py --checkpoint <checkpoint_file> --images <image_dir> --output <output_dir>
"""
import os
import sys
import argparse
import cv2
import glob
from pathlib import Path
import torch
import numpy as np

# Import mmyolo first to register custom modules
import yolo_world  # noqa: F401
from mmengine.config import Config
from mmdet.apis import init_detector


def parse_args():
    parser = argparse.ArgumentParser(description='OWOD Inference and Visualization')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file path')
    parser.add_argument('--config', default='configs/owod_ft/yolo_uniow_s_lora_bn_1e-3_20e_8gpus_owod.py', 
                        help='Config file path')
    parser.add_argument('--images', default='./images', help='Image directory or single image path')
    parser.add_argument('--output', default='./outputs', help='Output directory for results')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Score threshold')
    parser.add_argument('--max-dets', type=int, default=100, help='Max detections')
    args = parser.parse_args()
    return args


def inference(model, image_path, device='cuda:0', score_thr=0.3, max_dets=100):
    """Run inference on a single image."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None
    
    original_img = img.copy()
    h, w = img.shape[:2]
    
    # Prepare input
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Normalize
    img_tensor = img_tensor / 255.0
    
    # Inference
    with torch.no_grad():
        # Create data batch
        data_batch = {
            'inputs': img_tensor,
        }
        
        # Forward pass
        if hasattr(model, 'test_step'):
            results = model.test_step(data_batch)
            if isinstance(results, (list, tuple)):
                output = results[0]
            else:
                output = results
        else:
            output = model(img_tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]
        
        pred_instances = output.pred_instances
    
    if len(pred_instances) == 0:
        return {
            'image': original_img,
            'boxes': np.array([]),
            'labels': np.array([]),
            'scores': np.array([])
        }
    
    # Filter by score threshold
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    
    # Keep only top-k detections
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]
    
    pred_instances_numpy = pred_instances.cpu().numpy()
    
    boxes = pred_instances_numpy['bboxes'] if 'bboxes' in pred_instances_numpy else np.array([])
    scores = pred_instances_numpy['scores'] if 'scores' in pred_instances_numpy else np.array([])
    
    return {
        'image': original_img,
        'boxes': boxes,
        'scores': scores
    }


def draw_results(result, output_path):
    """Draw detection results on image and save."""
    img = result['image'].copy()
    boxes = result['boxes']
    scores = result['scores']
    
    h, w = img.shape[:2]
    
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(1, min(x2, w-1))
        y2 = max(1, min(y2, h-1))
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw score
        text = f'{score:.2f}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img, (x1, y1-25), (x1+text_size[0]+5, y1), (0, 255, 0), -1)
        cv2.putText(img, text, (x1+2, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Save result
    cv2.imwrite(output_path, img)
    print(f"✓ Saved result to: {output_path}")
    
    return img


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Change to project root directory for embeddings loading
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Load model
    print(f"Loading checkpoint from: {args.checkpoint}")
    print(f"Loading config from: {args.config}")
    
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    
    # Build model directly without creating test_dataset
    from mmyolo.registry import MODELS
    model = MODELS.build(cfg.model)
    
    # Load checkpoint
    from mmengine.runner import load_checkpoint
    load_checkpoint(model, args.checkpoint, map_location=args.device)
    
    model.to(args.device)
    model.eval()
    
    print(f"✓ Model loaded on device: {args.device}")
    
    # Get image files
    image_path = Path(args.images)
    if image_path.is_file():
        image_files = [str(image_path)]
    else:
        image_files = sorted(
            list(glob.glob(os.path.join(args.images, '*.jpg'))) + \
            list(glob.glob(os.path.join(args.images, '*.png'))) + \
            list(glob.glob(os.path.join(args.images, '*.jpeg')))
        )
    
    if not image_files:
        print(f"No images found in: {args.images}")
        return
    
    print(f"Found {len(image_files)} image(s)")
    
    # Inference
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_file}")
        
        result = inference(model, image_file, device=args.device,
                          score_thr=args.score_thr, max_dets=args.max_dets)
        
        if result is not None:
            num_dets = len(result['boxes'])
            print(f"  Detected {num_dets} object(s)")
            if num_dets > 0:
                print(f"  Scores: {result['scores']}")
            
            # Save result
            output_name = Path(image_file).stem + '_result.jpg'
            output_path = os.path.join(args.output, output_name)
            draw_results(result, output_path)
    
    print(f"\n✓ All processing complete! Results saved to: {args.output}")


if __name__ == '__main__':
    main()
