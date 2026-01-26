"""
Convert Locount dataset to PASCAL VOC format for OWOD training.

Locount format: x1,y1,x2,y2,class_name,count
VOC format: XML with bounding boxes

Usage:
    python tools/dataset_converters/locount_to_voc.py
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from tqdm import tqdm
import shutil
from PIL import Image


def create_voc_xml(image_path, boxes, output_path):
    """
    Create a VOC format XML file.
    
    Args:
        image_path: Path to the image file
        boxes: List of tuples (x1, y1, x2, y2, class_name)
        output_path: Path to save the XML file
    """
    # Get image dimensions
    with Image.open(image_path) as img:
        width, height = img.size
        depth = len(img.getbands())  # 1 for grayscale, 3 for RGB
    
    # Create root element
    annotation = ET.Element('annotation')
    
    # Add folder
    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'LocountOWOD'
    
    # Add filename
    filename = ET.SubElement(annotation, 'filename')
    filename.text = os.path.basename(image_path)
    
    # Add path
    path = ET.SubElement(annotation, 'path')
    path.text = str(image_path)
    
    # Add source
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Locount'
    
    # Add size
    size = ET.SubElement(annotation, 'size')
    width_elem = ET.SubElement(size, 'width')
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, 'height')
    height_elem.text = str(height)
    depth_elem = ET.SubElement(size, 'depth')
    depth_elem.text = str(depth)
    
    # Add segmented
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'
    
    # Add objects
    for x1, y1, x2, y2, class_name in boxes:
        obj = ET.SubElement(annotation, 'object')
        
        name = ET.SubElement(obj, 'name')
        name.text = class_name
        
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = '0'
        
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = '0'
        
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(int(x1) + 1)  # VOC format is 1-indexed
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(int(y1) + 1)
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(int(x2) + 1)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(int(y2) + 1)
    
    # Pretty print XML
    xml_str = ET.tostring(annotation, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent='  ')
    
    # Remove extra blank lines
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    pretty_xml = '\n'.join(lines)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)


def convert_locount_to_voc(locount_root, owod_root, only_stems=None):
    """
    Convert Locount dataset to VOC format.
    
    Args:
        locount_root: Path to Locount dataset root
        owod_root: Path to OWOD data root
    """
    locount_root = Path(locount_root)
    owod_root = Path(owod_root)
    
    # Create output directories
    output_img_dir = owod_root / 'JPEGImages' / 'LocountOWOD'
    output_ann_dir = owod_root / 'Annotations' / 'LocountOWOD'
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_ann_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train and test splits
    splits = {
        'train': {
            'label_dir': locount_root / 'Locount_GtTxtsTrain' / 'Locount_GtTxtsTrain',
            'image_dir': locount_root / 'Locount_ImagesTrain' / 'Locount_ImagesTrain'
        },
        'test': {
            'label_dir': locount_root / 'Locount_GtTxtsTest' / 'Locount_GtTxtsTest',
            'image_dir': locount_root / 'Locount_ImagesTest' / 'Locount_ImagesTest'
        }
    }
    
    total_converted = 0
    total_errors = 0
    
    only_set = set(only_stems) if only_stems else None

    for split_name, paths in splits.items():
        label_dir = paths['label_dir']
        image_dir = paths['image_dir']
        
        if not label_dir.exists():
            print(f"Warning: {label_dir} does not exist, skipping {split_name}")
            continue
        
        print(f"\nProcessing {split_name} split...")
        label_files = sorted(label_dir.glob('*.txt'))
        
        for label_file in tqdm(label_files, desc=f"Converting {split_name}"):
            stem = label_file.stem

            if only_set and stem not in only_set:
                continue
            
            # Find image file
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                candidate = image_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if image_path is None:
                print(f"Warning: Image not found for {stem}")
                total_errors += 1
                continue
            
            # Parse label file
            boxes = []
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split(',')
                        if len(parts) < 6:
                            continue
                        
                        x1 = float(parts[0])
                        y1 = float(parts[1])
                        x2 = float(parts[2])
                        y2 = float(parts[3])
                        class_name = ','.join(parts[4:-1]).strip()
                        # count = int(parts[-1])  # We don't use count in VOC format
                        
                        boxes.append((x1, y1, x2, y2, class_name))
            except Exception as e:
                print(f"Error parsing {label_file}: {e}")
                total_errors += 1
                continue
            
            if not boxes:
                # keep empty annotations so image lists stay aligned
                print(f"Warning: No valid boxes in {label_file}, writing empty annotation")
            
            # Copy image
            output_img_path = output_img_dir / f"{stem}.jpg"
            try:
                # Convert to RGB if needed and save as JPEG
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(output_img_path, 'JPEG', quality=95)
            except Exception as e:
                print(f"Error copying image {image_path}: {e}")
                total_errors += 1
                continue
            
            # Create XML annotation
            output_xml_path = output_ann_dir / f"{stem}.xml"
            try:
                create_voc_xml(output_img_path, boxes, output_xml_path)
                total_converted += 1
            except Exception as e:
                print(f"Error creating XML for {stem}: {e}")
                total_errors += 1
                continue
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Total converted: {total_converted}")
    print(f"Total errors: {total_errors}")
    print(f"Output directories:")
    print(f"  Images: {output_img_dir}")
    print(f"  Annotations: {output_ann_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Locount to VOC format')
    parser.add_argument('--locount-root', type=str, 
                        default='Locount',
                        help='Path to Locount dataset root')
    parser.add_argument('--owod-root', type=str,
                        default='data/OWOD',
                        help='Path to OWOD data root')
    parser.add_argument('--only', type=str, nargs='*',
                        help='Optional list of image stems to convert (e.g., 022101)')
    args = parser.parse_args()
    
    convert_locount_to_voc(args.locount_root, args.owod_root, args.only)
