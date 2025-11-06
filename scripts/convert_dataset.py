"""
Convert YOLO format dataset to MediaPipe Model Maker format.
Organizes images into folders by gesture label.
"""

import shutil
from pathlib import Path
import yaml


def convert_dataset(yolo_dataset_path, output_path='gesture_dataset'):
    yolo_path = Path(yolo_dataset_path)
    output_path = Path(output_path)

    # Read class names from data.yaml
    with open(yolo_path / 'data.yaml', 'r') as f:
        data = yaml.safe_load(f)
        class_names = data['names']

    print(f"Converting dataset for {len(class_names)} gestures: {class_names}")
    print(f"Output directory: {output_path}\n")

    # Process train, valid, and test splits
    for split in ['train', 'valid', 'test']:
        split_output = output_path / split
        split_output.mkdir(parents=True, exist_ok=True)

        # Create folders for each gesture
        for class_name in class_names:
            (split_output / class_name).mkdir(exist_ok=True)

        # Create 'none' folder (required by Model Maker)
        (split_output / 'none').mkdir(exist_ok=True)

        images_dir = yolo_path / split / 'images'
        labels_dir = yolo_path / split / 'labels'

        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping {split}")
            continue

        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

        copied_count = {name: 0 for name in class_names}
        none_count = 0

        for img_path in image_files:
            label_path = labels_dir / (img_path.stem + '.txt')

            if label_path.exists() and label_path.stat().st_size > 0:
                # Read class from label file (YOLO format: class_id x y w h)
                with open(label_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        class_id = int(line.split()[0])
                        class_name = class_names[class_id]

                        dest_path = split_output / class_name / img_path.name
                        shutil.copy2(img_path, dest_path)
                        copied_count[class_name] += 1
            else:
                # No label - put in 'none' folder
                dest_path = split_output / 'none' / img_path.name
                shutil.copy2(img_path, dest_path)
                none_count += 1

        print(f"{split.upper()} Split:")
        for class_name in class_names:
            print(f"  {class_name}: {copied_count[class_name]} images")
        print(f"  none: {none_count} images")
        print()

    print(f"Dataset conversion complete! Output: {output_path}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python convert_dataset.py <path_to_yolo_dataset>")
        sys.exit(1)

    yolo_dataset = Path(sys.argv[1])
    convert_dataset(yolo_dataset, output_path='data/gesture_dataset')
