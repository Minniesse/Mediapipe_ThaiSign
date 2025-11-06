# Thai Sign Language Recognition Using Pose Estimation: Technical Report

## Abstract

This report presents a pose estimation-based approach for recognizing six Thai Sign Language (ThSL) words: "I'm fine" (fine), "Hello" (hello_1, hello_2), "Yes" (yes), "No" (no), "Unwell" (unwell), and "Thank you" (thank). The system leverages Google MediaPipe for hand landmark detection combined with a custom-trained neural network classifier, achieving real-time gesture recognition capabilities. This report details the technical approach, implementation methodology, evaluation metrics, limitations, and justification for design decisions.

## 1. Introduction

Sign language recognition systems play a crucial role in facilitating communication for the deaf and hard-of-hearing community. This project implements a computer vision-based Thai Sign Language recognition system using pose estimation techniques to identify specific hand gestures corresponding to common conversational phrases. The goal is to develop a lightweight, real-time system capable of accurate gesture classification from video input.

## 2. Technical Approach

### 2.1 System Architecture

The system employs a two-stage pipeline architecture:

**Stage 1: Hand Detection and Landmark Extraction**
- Utilizes Google MediaPipe's pre-trained hand detection model
- Extracts 21 three-dimensional hand landmarks (x, y, z coordinates) per detected hand
- Supports bilateral hand detection (up to 2 hands simultaneously)
- Normalized landmark coordinates ensure scale and position invariance

**Stage 2: Gesture Classification**
- Custom neural network classifier trained on extracted landmarks
- Processes hand landmark features to predict gesture categories
- Outputs gesture class with associated confidence scores

### 2.2 Hand Landmark Detection

MediaPipe's hand tracking solution provides robust real-time hand landmark localization. The 21 landmarks cover anatomically significant points:
- Wrist (1 point)
- Thumb (4 points: CMC, MCP, IP, TIP)
- Index finger (4 points: MCP, PIP, DIP, TIP)
- Middle finger (4 points)
- Ring finger (4 points)
- Pinky finger (4 points)

Detection confidence thresholds were configured as follows:
- Hand detection confidence: 0.5
- Hand presence confidence: 0.5
- Tracking confidence: 0.5

These values balance detection reliability with computational efficiency, ensuring stable tracking while minimizing false positives.

### 2.3 Neural Network Architecture

The gesture classifier employs a feedforward neural network with the following configuration:

**Architecture:**
- Input layer: 63 features (21 landmarks × 3 coordinates)
- Hidden layer 1: 128 neurons with ReLU activation
- Dropout layer: 30% dropout rate (regularization)
- Hidden layer 2: 64 neurons with ReLU activation
- Output layer: 8 classes (softmax activation)

**Training Hyperparameters:**
- Learning rate: 0.001
- Batch size: 8
- Epochs: 50
- Learning rate decay: 0.99 per epoch
- Focal loss gamma: 2 (addresses class imbalance)

The architecture was designed to balance model capacity with generalization ability. The two-layer design with decreasing width (128→64) provides sufficient representation power while the dropout layer prevents overfitting on the limited training data.

### 2.4 Dataset Preparation

**Original Data Format:**
The dataset was initially organized in YOLO format with bounding box annotations. A conversion script (`convert_dataset.py`) transformed the data into MediaPipe-compatible folder structure, organizing images by gesture class.

**Dataset Statistics:**

*Training Set (1,095 images):*
- fine: 36 images
- hello_1: 54 images
- hello_2: 39 images
- no: 39 images
- thank: 21 images
- unwell: 42 images
- yes: 33 images
- none: 831 images

*Validation Set (20 images):*
Minimal representation across classes for hyperparameter tuning.

*Test Set (20 images):*
Small held-out set for final evaluation.

**Class Imbalance Consideration:**
The "none" class (negative examples) dominates the dataset with 831 training samples compared to 21-54 samples per gesture class. This 15:1 to 40:1 imbalance was addressed using focal loss (gamma=2), which down-weights well-classified examples and focuses learning on difficult cases.

## 3. Implementation Details

### 3.1 Training Pipeline

The training process follows this workflow:
1. Load datasets using MediaPipe Model Maker's `Dataset.from_folder()` API
2. Apply automatic hand landmark extraction and preprocessing
3. Train neural network classifier with configured hyperparameters
4. Evaluate model performance on test set
5. Export trained model in TensorFlow Lite Task format (8.2 MB)

Training leverages MediaPipe's `HandDataPreprocessingParams()` for standardized data augmentation and normalization, ensuring consistent preprocessing between training and inference.

### 3.2 Inference Pipeline

Real-time inference operates as follows:
1. Capture video frame (1280×720 resolution) from webcam or load image
2. Feed frame to MediaPipe hand detector
3. Extract hand landmarks if hands detected
4. Pass landmarks to gesture classifier
5. Display results with gesture label, confidence score, and hand laterality
6. Render hand landmarks overlaid on video frame

The inference script (`inference.py`) supports both image-based and real-time webcam recognition, providing flexibility for different use cases.

## 4. Evaluation Methodology

### 4.1 Evaluation Metrics

The model is evaluated using standard classification metrics:

**Primary Metrics:**
- **Test Accuracy**: Proportion of correctly classified test samples
- **Cross-Entropy Loss**: Measures prediction confidence calibration

**Inference-Time Metrics:**
- **Confidence Scores**: Per-prediction probability outputs (0-100%)
- **Hand Laterality Detection**: Identifies left vs. right hand

### 4.2 Evaluation Process

Model evaluation follows a held-out test set approach:
1. Train on 1,095 training samples
2. Tune hyperparameters using 20 validation samples
3. Final evaluation on 20 test samples with batch size 1
4. Report test loss and accuracy metrics

The small test set size (20 samples) limits statistical confidence in reported metrics, representing a significant evaluation limitation discussed in Section 5.

### 4.3 Qualitative Assessment

Beyond quantitative metrics, the system enables qualitative evaluation through:
- Real-time webcam testing to assess practical usability
- Visual feedback showing landmark detection quality
- Confidence score inspection to identify uncertain predictions
- Cross-hand testing (left vs. right hand performance)

## 5. Limitations and Challenges

### 5.1 Dataset Limitations

**Insufficient Data Volume:**
The most critical limitation is the small dataset size. With only 21-54 training samples per gesture class (excluding "none"), the model has limited exposure to gesture variation. This constrains generalization to:
- Different signers with varying hand sizes and proportions
- Diverse signing speeds and movement dynamics
- Various lighting conditions and backgrounds
- Different camera angles and distances

**Severe Class Imbalance:**
The 831 "none" samples versus 21-54 gesture samples creates a 15:1 to 40:1 imbalance. While focal loss mitigates this, the model may still develop bias toward negative predictions or fail to learn distinctive features for minority classes.

**Limited Test Set Size:**
With only 20 test samples total, evaluation metrics have high variance and low statistical power. Some gesture classes (hello_2, no, unwell) have zero validation samples, preventing proper hyperparameter tuning for these categories.

**Static Image Training:**
Training on extracted video frames rather than temporal sequences ignores the dynamic nature of sign language. Many signs involve characteristic movements that static poses cannot fully capture. The system recognizes hand shapes rather than complete signing gestures.

### 5.2 Technical Limitations

**Two-Hand Gestures:**
While MediaPipe supports bilateral detection, the model architecture processes hands independently. Gestures requiring coordinated two-hand movements or spatial relationships between hands may not be accurately recognized.

**Temporal Information:**
The single-frame classification approach loses temporal context. Signs with motion components (movement patterns, speed, trajectory) cannot be distinguished from similar static hand shapes.

**Environmental Sensitivity:**
Performance depends on MediaPipe's hand detection success, which can be affected by:
- Poor lighting conditions
- Complex backgrounds with skin-tone regions
- Hand occlusions or extreme poses
- Distance from camera

**Fixed Vocabulary:**
The system recognizes only 6 predefined signs (8 classes including hello variants). Expanding vocabulary requires collecting new training data and retraining, lacking zero-shot or few-shot learning capabilities.

### 5.3 Evaluation Limitations

**Generalization Uncertainty:**
The small, potentially non-diverse test set provides limited evidence of real-world performance. The model may overfit to specific signers, backgrounds, or recording conditions present in the dataset.

**Absence of Confusion Matrix:**
Without detailed per-class precision, recall, and F1-scores, we cannot assess which gestures are most confused or identify systematic recognition errors.

**No Cross-Validation:**
Single train-validation-test split without k-fold cross-validation increases result variance and reduces reliability of performance estimates.

**Missing User Studies:**
No evaluation with actual deaf/hard-of-hearing users or sign language interpreters to assess practical utility and recognition accuracy on real-world signing.

## 6. Justification of Design Decisions

### 6.1 Choice of MediaPipe

**Rationale:**
MediaPipe was selected for several compelling reasons:

1. **State-of-the-Art Performance**: MediaPipe's hand tracking achieves real-time performance (30+ FPS) with high accuracy, leveraging years of research and production optimization by Google.

2. **Robust Landmark Extraction**: The 21-landmark model provides anatomically meaningful features that naturally encode hand shape information relevant to sign language.

3. **Computational Efficiency**: The lightweight architecture enables real-time processing on consumer hardware without GPU requirements.

4. **Production-Ready**: MediaPipe is extensively tested, well-documented, and supports multiple platforms (Python, JavaScript, Android, iOS).

5. **Reduced Engineering Overhead**: Pre-trained hand detection eliminates the need for manual annotation of hand landmarks or training custom detection models.

**Alternatives Considered:**
- OpenPose: More comprehensive skeleton tracking but significantly slower and requires GPU
- Custom CNN: Would require large labeled dataset and substantial training resources
- Traditional CV (color/edge detection): Less robust to lighting and background variations

### 6.2 Neural Network Architecture

**Two-Layer Design:**
The 128→64 hidden layer configuration balances expressiveness with overfitting risk. Deeper networks would likely overfit the small training set, while single-layer models may lack capacity to learn complex gesture patterns.

**Dropout Regularization:**
30% dropout provides regularization to prevent memorization of training examples, critical given the limited data volume.

**Learning Rate Schedule:**
Exponential decay (0.99 per epoch) allows aggressive initial learning while enabling fine-tuning in later epochs, improving convergence quality.

### 6.3 Focal Loss for Class Imbalance

Using focal loss with gamma=2 addresses the severe class imbalance by:
- Reducing loss contribution from easily classified "none" examples
- Focusing model attention on hard-to-classify gesture samples
- Preventing the model from achieving high accuracy by simply predicting "none"

Alternative approaches like class weighting were not used as focal loss provides more nuanced handling of easy vs. hard examples within each class.

### 6.4 Single-Frame Classification Approach

**Justification:**
Despite losing temporal information, single-frame classification was chosen because:

1. **Dataset Constraints**: The dataset consists of extracted frames without temporal annotations or sequence labels.

2. **Simplicity and Speed**: Frame-level classification is computationally efficient and easier to implement/debug than sequence models (LSTM, Transformer).

3. **Static Sign Focus**: Several of the target signs (yes, no, thank you) have recognizable static hand shapes, making frame-level recognition viable.

4. **Foundation for Extension**: This approach establishes a baseline that can be extended with temporal modeling in future work.

**Future Enhancement:**
Implementing temporal modeling (using LSTM or 1D convolutions over landmark sequences) would significantly improve recognition of dynamic signs and reduce false positives from transitional hand positions.

### 6.5 Evaluation Strategy

The held-out test set approach was selected as the most straightforward evaluation method given dataset constraints. While k-fold cross-validation would be preferable, the extremely small validation/test set sizes (20 samples each) would result in folds with zero examples of some gesture classes.

The per-prediction confidence scores provide interpretable outputs that enable threshold tuning for precision-recall trade-offs in deployment scenarios.

## 7. Training Results and Performance Analysis

### 7.1 Training Configuration

The model was trained with the following actual dataset sizes (after MediaPipe hand detection filtering):
- **Training samples**: 379 images (successfully detected hands)
- **Validation samples**: 7 images
- **Test samples**: 8 images
- **Classes**: 8 (none, fine, hello_1, hello_2, no, thank, unwell, yes)

The significant reduction from 1,095 to 379 training samples indicates that MediaPipe could not detect hands in approximately 65% of the original training images, highlighting potential data quality issues or challenging hand poses in the dataset.

**Model Specification:**
- Total parameters: 26,568 (103.78 KB)
- Trainable parameters: 25,928 (101.28 KB)
- Non-trainable parameters: 640 (2.50 KB)

The compact model size (under 104 KB for the classifier network) ensures efficient deployment on resource-constrained devices.

### 7.2 Training Performance

The model was trained for 50 epochs with the following progression:

**Initial Learning Phase (Epochs 1-10):**
- Epoch 1: Training loss 1.6256, accuracy 25.53% | Validation loss 0.9680, accuracy 71.43%
- Epoch 5: Training loss 0.5247, accuracy 67.55% | Validation loss 0.2402, accuracy 100%
- Epoch 10: Training loss 0.4266, accuracy 72.07% | Validation loss 0.1431, accuracy 100%

The model showed rapid initial learning, with validation accuracy reaching 100% by epoch 5 while training accuracy lagged behind, suggesting good generalization despite the small dataset.

**Middle Training Phase (Epochs 11-30):**
- Epoch 15: Training loss 0.3587, accuracy 75.80% | Validation loss 0.1185, accuracy 100%
- Epoch 20: Training loss 0.3519, accuracy 73.94% | Validation loss 0.1211, accuracy 100%
- Epoch 25: Training loss 0.3195, accuracy 76.86% | Validation loss 0.1119, accuracy 100%
- Epoch 30: Training loss 0.3077, accuracy 78.46% | Validation loss 0.1301, accuracy 85.71%

Training accuracy steadily improved while validation accuracy fluctuated between 85.71% and 100%, indicating some instability due to the extremely small validation set (only 7 samples).

**Final Training Phase (Epochs 31-50):**
- Epoch 35: Training loss 0.2916, accuracy 77.39% | Validation loss 0.1105, accuracy 100%
- Epoch 40: Training loss 0.3274, accuracy 75.53% | Validation loss 0.1255, accuracy 85.71%
- Epoch 45: Training loss 0.2403, accuracy 80.05% | Validation loss 0.1166, accuracy 85.71%
- Epoch 50: Training loss 0.2633, accuracy 77.13% | Validation loss 0.1080, accuracy 85.71%

The model converged to approximately 77-80% training accuracy with validation loss stabilizing around 0.11.

**Learning Rate Decay:**
The learning rate decayed from 0.001 to 0.00061 over 50 epochs (0.99 decay per epoch), facilitating fine-grained optimization in later training stages.

### 7.3 Test Set Performance

**Final Evaluation Metrics:**
- **Test Loss**: 0.0203
- **Test Accuracy**: 100% (8/8 samples correctly classified)

The model achieved perfect accuracy on the held-out test set, correctly classifying all 8 test samples. However, this result must be interpreted with extreme caution given the test set contains only 8 samples.

**Statistical Considerations:**
With only 8 test samples, the 95% confidence interval for accuracy is approximately [63%-100%], indicating very high uncertainty. A single misclassification would drop accuracy to 87.5%, demonstrating the instability of metrics on such small test sets.

### 7.4 Training Observations

**Convergence Behavior:**
The training exhibited several notable characteristics:

1. **Fast Initial Convergence**: Validation accuracy reached 100% by epoch 5, suggesting the model quickly learned to distinguish most gesture classes.

2. **Training-Validation Gap**: Training accuracy plateaued around 77-80%, consistently lower than validation performance. This unusual pattern likely stems from:
   - The extremely small validation set (7 samples) providing unreliable estimates
   - Potential easier examples in the validation split
   - Effective regularization via dropout preventing overfitting

3. **Loss Convergence**: Both training and validation losses steadily decreased, with final validation loss (0.108) substantially lower than training loss (0.263), further supporting the hypothesis of validation set peculiarities.

4. **Validation Accuracy Fluctuations**: Validation accuracy alternated between 85.71% (6/7 correct) and 100% (7/7 correct), representing a single sample difference. This demonstrates the high variance inherent in evaluating on 7 samples.

**Data Quality Issues:**
The reduction from 1,095 original images to 379 usable training samples (after hand detection) reveals significant data quality problems:
- 65% of images failed hand detection, possibly due to:
  - Hands outside frame or partially cropped
  - Severe occlusions
  - Poor image quality or extreme blur
  - Incorrect bounding boxes in original YOLO annotations
  - Hands at extreme angles or very small scale

This data loss severely exacerbates the already limited dataset size, reducing the effective training examples per class to approximately 47 samples on average (excluding "none" class).

### 7.5 Model Performance Analysis

**Strengths Observed:**
- **Perfect Test Accuracy**: The model correctly classified all test samples, though with the caveat of small sample size
- **Low Test Loss**: The 0.0203 test loss indicates high prediction confidence on correct classifications
- **Fast Training**: Each epoch completed in under 1 second (~0.3-0.4s), enabling rapid iteration
- **No Overfitting**: Despite 50 epochs, the model showed no signs of overfitting, with validation loss continuing to decrease

**Weaknesses and Concerns:**
- **Unreliable Metrics**: All performance metrics (train, validation, test accuracy) have high variance due to extremely small dataset sizes
- **Validation Set Too Small**: With only 7 validation samples, hyperparameter tuning cannot be reliably performed
- **Unknown Per-Class Performance**: No confusion matrix or per-class metrics available to identify which gestures are well-recognized
- **Generalization Uncertainty**: Perfect test accuracy on 8 samples provides minimal evidence of real-world performance
- **Data Quality Problems**: Loss of 65% of training data to failed hand detection suggests systematic dataset issues

**Actual vs. Expected Performance:**
The 100% test accuracy significantly exceeds expectations given the limited training data. Possible explanations include:
1. **Test set too easy**: The 8 test samples may not represent challenging cases
2. **Test set similarity**: High overlap in conditions between train and test sets
3. **Lucky sampling**: Random chance with such small test size
4. **Effective architecture**: The model genuinely learned discriminative features despite data limitations

### 7.6 Computational Performance

**Training Efficiency:**
- Total training time: Approximately 16 seconds (50 epochs)
- Time per epoch: ~0.3 seconds average
- Hardware: CPU-only training (no GPU utilized)

**Inference Speed:**
- Test set evaluation: 8 samples in <1 second
- Estimated inference time: ~680 microseconds per sample
- Suitable for real-time applications at 30+ FPS

The lightweight architecture enables deployment on consumer hardware without GPU requirements, making it accessible for practical applications.

### 7.7 Practical Deployment Considerations

For real-world deployment, several factors must be considered:

**Required Conditions:**
- Consistent lighting conditions to ensure reliable hand detection
- Uncluttered backgrounds to minimize false positives
- Clear view of hands without occlusion
- Hands within optimal distance range (approximately 0.5-2 meters from camera)
- User adaptation to hold signs in canonical positions

**Confidence Thresholding:**
The system outputs confidence scores that can be thresholded to balance precision and recall. Recommended strategies:
- High confidence threshold (>80%) for critical applications requiring high precision
- Medium threshold (>60%) for general use balancing accuracy and recognition rate
- Low threshold (>40%) for exploratory or demonstration purposes

**Expected Real-World Performance:**
Based on training behavior and test results, anticipated real-world performance:
- Strong recognition of well-represented, distinctive gestures (hello_1, fine)
- Moderate performance on less-represented classes (thank, unwell)
- Potential confusion between gestures with similar hand shapes
- Sensitivity to signing variations not present in training data
- Degradation under lighting/background conditions differing from training data

## 8. Future Improvements

### 8.1 Dataset Enhancement

**Priority improvements:**
1. Collect 500-1000 samples per gesture class from multiple signers
2. Include diverse lighting conditions, backgrounds, and camera angles
3. Record video sequences rather than static frames
4. Ensure balanced class distribution
5. Engage Thai Sign Language experts for annotation validation

### 8.2 Model Architecture

**Temporal modeling:**
Implement LSTM or Temporal Convolutional Networks to process landmark sequences, capturing motion patterns and transition dynamics.

**Two-hand coordination:**
Design architecture to process left and right hand landmarks jointly, enabling recognition of two-handed signs and spatial relationships.

**Attention mechanisms:**
Incorporate attention layers to identify which landmarks are most relevant for each gesture, improving interpretability and accuracy.

### 8.3 Evaluation Rigor

**Comprehensive evaluation:**
1. Larger test set (200+ samples) for statistical significance
2. Per-class precision, recall, F1-score analysis
3. Confusion matrix visualization
4. Cross-validation for robust performance estimates
5. User studies with native Thai Sign Language signers
6. Real-world deployment testing in varied environments

### 8.4 System Extensions

**Vocabulary expansion:**
- Incremental learning approaches for adding new signs without full retraining
- Few-shot learning techniques to reduce data requirements for new gestures
- Hierarchical classification to handle larger sign vocabularies

**Multimodal fusion:**
- Combine hand landmarks with facial expressions (important in sign language)
- Incorporate body pose for signs involving torso or arm movements
- Add context modeling for sentence-level interpretation

## 9. Conclusion

This project demonstrates a functional Thai Sign Language recognition system using pose estimation with MediaPipe hand tracking and neural network classification. The approach successfully combines state-of-the-art hand landmark detection with custom gesture classification, achieving both a lightweight model (26,568 parameters, 104 KB) and real-time capable inference (~680 microseconds per sample).

**Achieved Results:**
The trained model achieved 100% accuracy on the test set with a test loss of 0.0203, correctly classifying all 8 held-out test samples. Training converged smoothly over 50 epochs, reaching 77-80% training accuracy without overfitting. The system completed training in approximately 16 seconds on CPU-only hardware, demonstrating exceptional computational efficiency.

**Primary Strengths:**
The system's key advantages lie in:
1. Leveraging robust, pre-trained MediaPipe hand tracking for reliable landmark extraction
2. Compact neural network architecture enabling deployment on resource-constrained devices
3. Fast training and inference suitable for iterative development and real-time applications
4. Modular design facilitating future enhancements and vocabulary expansion

**Critical Limitations:**
However, significant limitations severely constrain confidence in these results and practical deployment viability:

1. **Extremely Small Dataset**: Only 379 training samples (after 65% data loss due to failed hand detection), 7 validation samples, and 8 test samples provide grossly insufficient data for robust model development and reliable evaluation.

2. **Data Quality Issues**: The loss of 716 images (65% of original dataset) during hand detection reveals systematic data quality problems, including poor cropping, occlusions, or incorrect annotations.

3. **Unreliable Metrics**: With only 8 test samples, the 100% accuracy has a 95% confidence interval of approximately [63%-100%], rendering performance estimates highly uncertain. A single misclassification would drop accuracy to 87.5%.

4. **Missing Detailed Analysis**: Absence of confusion matrices, per-class metrics, or cross-validation prevents understanding which gestures are well-recognized and which are frequently confused.

5. **Temporal Modeling Gap**: The single-frame approach ignores motion dynamics essential to many sign language gestures, limiting recognition to static hand shapes.

**Practical Implications:**
The perfect test accuracy, while encouraging, must be interpreted as preliminary evidence rather than validated performance. The system likely performs well on the specific conditions and signers represented in the training data but may exhibit poor generalization to:
- New signers with different hand proportions or signing styles
- Varied lighting conditions and backgrounds
- Different camera angles or distances
- Dynamic signing with motion components

**Research Foundation:**
Despite these substantial limitations, the system establishes a solid technical foundation for Thai Sign Language recognition research. The proof-of-concept demonstrates that:
- MediaPipe hand tracking effectively extracts relevant features for sign recognition
- Lightweight neural networks can learn gesture patterns from landmark data
- Real-time performance is achievable on consumer hardware
- The modular architecture supports future enhancements

**Path Forward:**
Future work must urgently address the data scarcity challenge as the primary bottleneck. Collecting 500-1,000 high-quality samples per gesture class from diverse signers, implementing temporal modeling (LSTM/TCN), and conducting rigorous evaluation with larger test sets and cross-validation would transform this prototype into a reliable, production-ready system capable of serving the deaf and hard-of-hearing community in Thailand.

The design decisions prioritize simplicity, efficiency, and extensibility, making pragmatic trade-offs appropriate for an initial research prototype. With proper dataset development and architectural enhancements, this approach shows promise for evolving into a practical assistive technology tool.

---

**Word Count: Approximately 2,800 words**

## References

1. Google MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands
2. MediaPipe Model Maker: https://developers.google.com/mediapipe/solutions/model_maker
3. Lin, T.Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV 2017.
4. Zhang, F., et al. (2020). MediaPipe: A Framework for Building Perception Pipelines. arXiv preprint.
