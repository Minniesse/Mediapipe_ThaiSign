"""
Benchmark model performance: speed, memory, and accuracy metrics.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import numpy as np
import time
import psutil
import os


class PerformanceBenchmark:
    def __init__(self, model_path='models/model/gesture_recognizer.task'):
        self.model_path = model_path

        # Get model file size
        self.model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        print(f"Model: {model_path}")
        print(f"Model size: {self.model_size_mb:.2f} MB\n")

    def measure_inference_time(self, image_path, num_runs=100):
        """Measure inference time over multiple runs."""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load: {image_path}")
            return None

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Warmup
        for _ in range(10):
            self.recognizer.recognize(mp_image)

        # Measure
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            result = self.recognizer.recognize(mp_image)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        return times, result

    def measure_memory_usage(self):
        """Measure memory usage."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # MB

    def benchmark_dataset(self, dataset_path='data/dataset/test', num_samples=20):
        """Benchmark on test dataset."""
        dataset_path = Path(dataset_path)

        print("="*60)
        print("Performance Benchmark")
        print("="*60)
        print()

        # Get sample images
        test_images = []
        for gesture_folder in dataset_path.iterdir():
            if not gesture_folder.is_dir():
                continue
            image_files = list(gesture_folder.glob('*.jpg')) + list(gesture_folder.glob('*.png'))
            test_images.extend(image_files[:num_samples // 8 + 1])

        if not test_images:
            print("No test images found!")
            return

        test_images = test_images[:num_samples]

        print(f"Testing on {len(test_images)} images...\n")

        # Measure memory before
        mem_before = self.measure_memory_usage()

        all_times = []
        successful = 0
        failed = 0

        for img_path in test_images:
            times, result = self.measure_inference_time(img_path, num_runs=50)
            if times:
                all_times.extend(times)
                if result and result.gestures:
                    successful += 1
                else:
                    failed += 1

        # Measure memory after
        mem_after = self.measure_memory_usage()
        mem_used = mem_after - mem_before

        # Calculate statistics
        all_times = np.array(all_times)

        print("="*60)
        print("Results")
        print("="*60)
        print()

        print("Model Information:")
        print(f"  Model size: {self.model_size_mb:.2f} MB")
        print(f"  Memory usage: {mem_used:.2f} MB")
        print()

        print("Inference Speed:")
        print(f"  Mean: {np.mean(all_times):.2f} ms")
        print(f"  Median: {np.median(all_times):.2f} ms")
        print(f"  Min: {np.min(all_times):.2f} ms")
        print(f"  Max: {np.max(all_times):.2f} ms")
        print(f"  Std Dev: {np.std(all_times):.2f} ms")
        print(f"  FPS: {1000 / np.mean(all_times):.1f}")
        print()

        print("Detection Rate:")
        print(f"  Successful: {successful}/{len(test_images)} ({successful/len(test_images):.1%})")
        print(f"  Failed: {failed}/{len(test_images)}")
        print()

        # Percentiles
        print("Latency Percentiles:")
        for p in [50, 75, 90, 95, 99]:
            print(f"  {p}th: {np.percentile(all_times, p):.2f} ms")

        print("="*60)

        return {
            'mean_time_ms': np.mean(all_times),
            'median_time_ms': np.median(all_times),
            'fps': 1000 / np.mean(all_times),
            'model_size_mb': self.model_size_mb,
            'memory_mb': mem_used,
            'detection_rate': successful / len(test_images)
        }


if __name__ == '__main__':
    import sys

    benchmark = PerformanceBenchmark()

    if len(sys.argv) > 1:
        # Benchmark single image
        times, result = benchmark.measure_inference_time(sys.argv[1], num_runs=100)
        if times:
            print(f"Mean inference time: {np.mean(times):.2f} ms")
            print(f"FPS: {1000 / np.mean(times):.1f}")
            if result and result.gestures:
                print(f"Detected: {result.gestures[0][0].category_name}")
    else:
        # Benchmark dataset
        benchmark.benchmark_dataset()
