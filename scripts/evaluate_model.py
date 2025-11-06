"""
Evaluate model and generate confusion matrix visualization.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class ModelEvaluator:
    def __init__(self, model_path='models/model/gesture_recognizer.task'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        # Load labels
        labels_path = Path(model_path).parent / 'labels.txt'
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        print(f"Loaded model: {model_path}")
        print(f"Classes: {self.labels}\n")

    def evaluate_dataset(self, dataset_path='data/dataset/test'):
        """Evaluate model on test dataset and collect predictions."""
        dataset_path = Path(dataset_path)

        y_true = []
        y_pred = []

        print("Evaluating test dataset...")

        # Process each gesture folder
        for gesture_folder in sorted(dataset_path.iterdir()):
            if not gesture_folder.is_dir():
                continue

            gesture_name = gesture_folder.name
            if gesture_name not in self.labels:
                continue

            true_label = self.labels.index(gesture_name)

            # Process each image in the folder
            image_files = list(gesture_folder.glob('*.jpg')) + list(gesture_folder.glob('*.png'))

            for img_path in image_files:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                # Convert to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

                # Recognize
                result = self.recognizer.recognize(mp_image)

                if result.gestures:
                    predicted_gesture = result.gestures[0][0].category_name
                    predicted_label = self.labels.index(predicted_gesture)

                    y_true.append(true_label)
                    y_pred.append(predicted_label)

        return np.array(y_true), np.array(y_pred)

    def plot_confusion_matrix(self, y_true, y_pred, output_path='confusion_matrix.png'):
        """Generate and save confusion matrix visualization."""
        # Get unique labels present in test data
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        test_label_names = [self.labels[i] for i in unique_labels]

        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=test_label_names, yticklabels=test_label_names,
                    cbar_kws={'label': 'Count'})

        plt.title('Confusion Matrix - Thai Sign Language Gestures', fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {output_path}")
        plt.close()

    def print_metrics(self, y_true, y_pred):
        """Print classification metrics."""
        print("\n" + "="*60)
        print("Classification Report")
        print("="*60)

        # Get unique labels present in data
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        test_label_names = [self.labels[i] for i in unique_labels]

        print(classification_report(y_true, y_pred, labels=unique_labels,
                                   target_names=test_label_names, zero_division=0))

        accuracy = np.mean(y_true == y_pred)
        print(f"Overall Accuracy: {accuracy:.2%}")
        print("="*60)

    def evaluate(self, dataset_path='data/gesture_dataset/test', output_path='confusion_matrix.png'):
        """Run full evaluation pipeline."""
        y_true, y_pred = self.evaluate_dataset(dataset_path)

        if len(y_true) == 0:
            print("No test samples found!")
            return

        print(f"\nEvaluated {len(y_true)} samples")

        self.print_metrics(y_true, y_pred)
        self.plot_confusion_matrix(y_true, y_pred, output_path)


if __name__ == '__main__':
    import sys

    evaluator = ModelEvaluator()

    # Allow custom output path
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'confusion_matrix.png'

    evaluator.evaluate(output_path=output_path)
