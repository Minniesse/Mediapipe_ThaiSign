"""
Real-time hand gesture recognition using webcam or single images.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
from pathlib import Path


class GestureRecognizer:
    def __init__(self, model_path='models/model/gesture_recognizer.task'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        print(f"Loaded model: {model_path}\n")

    def recognize_image(self, image_path):
        """Recognize gesture from an image file."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return

        print(f"Testing: {image_path}\n")

        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run recognition
        result = self.recognizer.recognize(mp_image)

        if result.gestures:
            print(f"Detected {len(result.gestures)} hand(s):\n")

            for idx, (gestures, handedness) in enumerate(zip(result.gestures, result.handedness)):
                hand_label = handedness[0].category_name
                top_gesture = gestures[0]

                print(f"Hand {idx + 1} ({hand_label}):")
                print(f"  Gesture: {top_gesture.category_name}")
                print(f"  Confidence: {top_gesture.score:.2%}\n")
        else:
            print("No hands detected\n")

    def run_webcam(self):
        """Run real-time gesture recognition on webcam."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Starting webcam... Press 'q' to quit\n")

        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        from mediapipe.framework.formats import landmark_pb2

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            result = self.recognizer.recognize(mp_image)

            # Draw results
            if result.gestures and result.hand_landmarks:
                for idx, (gestures, landmarks, handedness) in enumerate(
                    zip(result.gestures, result.hand_landmarks, result.handedness)
                ):
                    # Convert landmarks to proper format for drawing
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                        for landmark in landmarks
                    ])

                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks_proto, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )

                    # Display gesture text
                    top_gesture = gestures[0]
                    hand_label = handedness[0].category_name
                    y_offset = 50 + idx * 60
                    text = f"{hand_label}: {top_gesture.category_name} ({top_gesture.score:.0%})"

                    # Background box
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    cv2.rectangle(image, (5, y_offset - 35), (tw + 15, y_offset + 10), (0, 255, 0), -1)

                    # Text
                    cv2.putText(image, text, (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            else:
                cv2.putText(image, "No hand detected", (10, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Hand Gesture Recognition', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    recognizer = GestureRecognizer()

    if len(sys.argv) > 1:
        # Test from image
        recognizer.recognize_image(sys.argv[1])
    else:
        # Run webcam
        recognizer.run_webcam()
