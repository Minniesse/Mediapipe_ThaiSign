"""
Train hand gesture recognizer using MediaPipe Model Maker.
"""

from mediapipe_model_maker import gesture_recognizer
import tensorflow as tf
from pathlib import Path


def train_model(dataset_path='data/dataset', output_dir='models/model'):
    print("Loading dataset...")

    # Load training data
    train_data = gesture_recognizer.Dataset.from_folder(
        dirname=f'{dataset_path}/train',
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
    print(f"Train samples: {len(train_data)}")

    # Load validation data
    validation_data = gesture_recognizer.Dataset.from_folder(
        dirname=f'{dataset_path}/valid',
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
    print(f"Validation samples: {len(validation_data)}")

    # Load test data
    test_data = gesture_recognizer.Dataset.from_folder(
        dirname=f'{dataset_path}/test',
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
    print(f"Test samples: {len(test_data)}")
    print(f"Classes: {train_data.label_names}\n")

    # Configure training parameters
    hparams = gesture_recognizer.HParams(
        export_dir=output_dir,
        learning_rate=0.001,
        batch_size=8,
        epochs=50,
        shuffle=True,
        lr_decay=0.99,
        gamma=2
    )

    model_options = gesture_recognizer.ModelOptions(
        dropout_rate=0.3,
        layer_widths=[128, 64]
    )

    options = gesture_recognizer.GestureRecognizerOptions(
        model_options=model_options,
        hparams=hparams
    )

    print("Training model...")
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options
    )

    print("\nEvaluating on test set...")
    loss, accuracy = model.evaluate(test_data, batch_size=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}\n")

    # Export model
    print("Exporting model...")
    model.export_model()
    print(f"Model saved to: {output_dir}/gesture_recognizer.task")

    # Save labels
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'labels.txt', 'w') as f:
        for label in train_data.label_names:
            f.write(f"{label}\n")

    print("Training complete!")


if __name__ == '__main__':
    tf.random.set_seed(42)
    train_model()
