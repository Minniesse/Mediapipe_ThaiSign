"""
Test model robustness with various image transformations.
Evaluates how the model handles different conditions.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class RobustnessTest:
    def __init__(self, model_path='models/model/gesture_recognizer.task'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        labels_path = Path(model_path).parent / 'labels.txt'
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        print("Testing model robustness...\n")

    def predict_image(self, image):
        """Predict gesture from numpy image."""
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self.recognizer.recognize(mp_image)

        if result.gestures:
            return result.gestures[0][0].category_name, result.gestures[0][0].score
        return None, 0.0

    # Image transformations
    def adjust_brightness(self, image, factor):
        """Adjust brightness."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def add_noise(self, image, noise_level=25):
        """Add Gaussian noise."""
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)

    def apply_blur(self, image, kernel_size=5):
        """Apply Gaussian blur."""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def rotate_image(self, image, angle):
        """Rotate image."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))

    def test_transformations(self, image_path):
        """Test various transformations on a single image."""
        original = cv2.imread(str(image_path))
        if original is None:
            print(f"Failed to load: {image_path}")
            return None

        # Get original prediction
        orig_pred, orig_conf = self.predict_image(original)
        if orig_pred is None:
            print(f"No hand detected in: {image_path}")
            return None

        print(f"Testing: {Path(image_path).name}")
        print(f"Original prediction: {orig_pred} ({orig_conf:.2%})\n")

        results = {'original': (orig_pred, orig_conf)}

        # Test brightness variations
        for factor in [0.5, 1.5, 2.0]:
            img = self.adjust_brightness(original.copy(), factor)
            pred, conf = self.predict_image(img)
            results[f'brightness_{factor}'] = (pred, conf)
            match = "✓" if pred == orig_pred else "✗"
            print(f"  Brightness {factor}x: {pred} ({conf:.2%}) {match}")

        # Test noise
        for noise in [10, 25, 50]:
            img = self.add_noise(original.copy(), noise)
            pred, conf = self.predict_image(img)
            results[f'noise_{noise}'] = (pred, conf)
            match = "✓" if pred == orig_pred else "✗"
            print(f"  Noise level {noise}: {pred} ({conf:.2%}) {match}")

        # Test blur
        for kernel in [3, 7, 11]:
            img = self.apply_blur(original.copy(), kernel)
            pred, conf = self.predict_image(img)
            results[f'blur_{kernel}'] = (pred, conf)
            match = "✓" if pred == orig_pred else "✗"
            print(f"  Blur kernel {kernel}: {pred} ({conf:.2%}) {match}")

        # Test rotation
        for angle in [-15, -30, 15, 30]:
            img = self.rotate_image(original.copy(), angle)
            pred, conf = self.predict_image(img)
            results[f'rotate_{angle}'] = (pred, conf)
            match = "✓" if pred == orig_pred else "✗"
            print(f"  Rotation {angle}°: {pred} ({conf:.2%}) {match}")

        print()
        return results

    def test_dataset(self, dataset_path='data/dataset/test', num_samples=5):
        """Test robustness on multiple images."""
        dataset_path = Path(dataset_path)
        all_results = {}

        print("="*60)
        print("Robustness Testing")
        print("="*60)
        print()

        count = 0
        for gesture_folder in dataset_path.iterdir():
            if not gesture_folder.is_dir():
                continue

            image_files = list(gesture_folder.glob('*.jpg')) + list(gesture_folder.glob('*.png'))
            for img_path in image_files[:num_samples]:
                results = self.test_transformations(img_path)
                if results:
                    all_results[img_path.name] = results
                    count += 1

        # Calculate statistics
        self.print_statistics(all_results)
        return all_results

    def print_statistics(self, all_results):
        """Print overall robustness statistics."""
        print("\n" + "="*60)
        print("Robustness Statistics")
        print("="*60)

        if not all_results:
            print("No results to analyze")
            return

        # Count matches for each transformation type
        transform_types = ['brightness', 'noise', 'blur', 'rotate']
        stats = {t: {'correct': 0, 'total': 0} for t in transform_types}

        for img_results in all_results.values():
            orig_pred = img_results['original'][0]

            for key, (pred, conf) in img_results.items():
                if key == 'original':
                    continue

                transform = key.split('_')[0]
                if transform in stats:
                    stats[transform]['total'] += 1
                    if pred == orig_pred:
                        stats[transform]['correct'] += 1

        print()
        for transform, counts in stats.items():
            if counts['total'] > 0:
                accuracy = counts['correct'] / counts['total']
                print(f"{transform.capitalize():12s}: {accuracy:.1%} "
                      f"({counts['correct']}/{counts['total']} consistent)")

        print("="*60)


if __name__ == '__main__':
    import sys

    tester = RobustnessTest()

    if len(sys.argv) > 1:
        # Test single image
        tester.test_transformations(sys.argv[1])
    else:
        # Test dataset
        tester.test_dataset(num_samples=3)
