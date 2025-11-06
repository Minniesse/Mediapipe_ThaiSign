# Thai Sign Language Recognition

Real-time Thai Sign Language (ThSL) gesture recognition system using MediaPipe and neural networks. Recognizes 7 common Thai signs with 90% accuracy at 68.6 FPS on CPU.

## Features

- **Real-time Performance**: 68.6 FPS on CPU (14.57ms inference)
- **Lightweight**: 8.16 MB model, 26,568 parameters
- **7 Gestures**: fine, hello_1, hello_2, no, thank, unwell, yes
- **High Accuracy**: 90% test accuracy
- **No GPU Required**: Runs on consumer hardware

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Real-time Webcam Inference

```bash
python scripts/inference.py
```

Press 'q' to quit.

### Single Image Test

```bash
python scripts/inference.py path/to/image.jpg
```

## Training

Train the model on your own dataset:

```bash
python scripts/train.py
```

The script will:
- Load data from `data/dataset/{train,valid,test}`
- Train for 50 epochs (~25 seconds on CPU)
- Export model to `models/model/gesture_recognizer.task`

## Evaluation

### Generate Confusion Matrix

```bash
python scripts/evaluate_model.py
```

Outputs classification report and saves confusion matrix to `confusion_matrix.png`.

### Performance Benchmark

```bash
python scripts/performance_benchmark.py
```

Measures:
- Inference speed (FPS, latency)
- Memory usage
- Detection rate

### Robustness Testing

```bash
python scripts/robustness_test.py
```

Tests model under:
- Brightness variations
- Gaussian noise
- Blur
- Rotation

## Project Structure

```
Mediapipe_ThaiSign/
├── data/
│   └── dataset/           # Training data (189 train, 37 valid, 50 test)
│       ├── train/
│       ├── valid/
│       └── test/
├── models/
│   └── model/             # Trained model
│       ├── gesture_recognizer.task
│       └── labels.txt
├── scripts/
│   ├── train.py           # Train model
│   ├── inference.py       # Real-time inference
│   ├── evaluate_model.py  # Generate confusion matrix
│   ├── performance_benchmark.py
│   └── robustness_test.py
├── results/               # Evaluation results
│   ├── confusion_matrix.png
│   ├── performance_benchmark.png
│   └── robustness_test.png
├── requirements.txt
├── TECHNICAL_REPORT.md    # Detailed technical report
└── README.md
```

## Model Architecture

- **Input**: 63 features (21 hand landmarks × 3 coordinates)
- **Hidden**: 128 → 64 neurons with ReLU
- **Dropout**: 30%
- **Output**: 7 gesture classes with Softmax
- **Parameters**: 26,568 (including batch normalization)

## Dataset

7 Thai Sign Language gestures collected from video:
- **fine**: I'm fine
- **hello_1**: Hello (variant 1)
- **hello_2**: Hello (variant 2)
- **no**: No
- **thank**: Thank you
- **unwell**: I'm not well
- **yes**: Yes

Split: 70% train / 15% valid / 15% test

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 90.00% |
| Inference Speed | 68.6 FPS (14.57ms) |
| Model Size | 8.16 MB |
| Memory Usage | 16.52 MB |
| Detection Rate | 85% |

### Robustness

- ✅ **Brightness**: 100% consistent (0.5x - 2.0x)
- ✅ **Blur**: 100% consistent (kernel 3-11)
- ⚠️ **Rotation**: 67% consistent (±15° good, ±30° struggles)
- ❌ **Noise**: 33% consistent (fails at level 25+)

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- TensorFlow Lite
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- psutil

## Technical Details

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for comprehensive analysis including:
- MediaPipe pipeline architecture
- Neural network design
- Training methodology
- Performance analysis
- Limitations and future work

## References

- [MediaPipe Hands Documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [MediaPipe Hands Paper](https://arxiv.org/abs/2006.10214) - Zhang et al. (2020)
- [TensorFlow Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker)
- [Thai Association of the Deaf](http://www.deafthai.or.th/)

## License

This project is for educational purposes.
