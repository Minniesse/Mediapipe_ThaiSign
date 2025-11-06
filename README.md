# Thai Sign Language Recognition with MediaPipe

Hand gesture recognition for Thai Sign Language using MediaPipe Model Maker.

## Project Structure

```
Mediapipe_ThaiSign/
├── scripts/
│   ├── convert_dataset.py    # Convert YOLO format to MediaPipe format
│   ├── train.py               # Train the gesture recognizer
│   └── inference.py           # Run inference on webcam or images
├── models/                    # Trained models
├── data/                      # Dataset folder
└── README.md
```

## Setup

Install dependencies:

```bash
uv add mediapipe mediapipe-model-maker opencv-python pyyaml
```

Or using pip:

```bash
pip install mediapipe mediapipe-model-maker opencv-python pyyaml
```

## Usage

### 1. Convert Dataset

Convert your YOLO format dataset to MediaPipe format:

```bash
python scripts/convert_dataset.py /path/to/yolo/dataset
```

This creates a `data/gesture_dataset` folder organized by gesture labels.

### 2. Train Model

Train the gesture recognizer:

```bash
python scripts/train.py
```

The trained model will be saved to `models/exported_model/gesture_recognizer.task`.

### 3. Run Inference

Test on an image:

```bash
python scripts/inference.py test_image.jpg
```

Or run real-time recognition with webcam:

```bash
python scripts/inference.py
```

Press 'q' to quit the webcam view.

## Gestures

The model recognizes these Thai Sign Language gestures:
- fine
- hello_1
- hello_2
- no
- thank
- unwell
- yes

## Notes

Dataset should be organized with images in folders by gesture name. MediaPipe will automatically extract hand landmarks during training.
