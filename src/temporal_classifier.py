from collections import deque
import pickle

import numpy as np
import tensorflow as tf


SEQUENCE_LENGTH = 60

ANGLE_FEATURES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "elbow_symmetry",
    "knee_symmetry",
    "shoulder_symmetry",
    "hip_symmetry",
    "torso_lean",
]

LANDMARK_NAMES = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

LANDMARK_FEATURES = [
    f"{name}_{axis}"
    for name in LANDMARK_NAMES
    for axis in ["x", "y", "z", "vis"]
]

FEATURES = ANGLE_FEATURES + LANDMARK_FEATURES


class TemporalClassifier:

    def __init__(
        self,
        model_path="models/exercise_cnn.keras",
        scaler_path="models/exercise_cnn_scaler.pkl",
        encoder_path="models/exercise_cnn_encoder.pkl",
    ):
        self.model = tf.keras.models.load_model(model_path)

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        with open(encoder_path, "rb") as f:
            self.encoder = pickle.load(f)

        self.buffer = deque(maxlen=SEQUENCE_LENGTH)

    def add_frame(self, features):
        """
        Add one frame of features.

        Returns:
            None until the buffer is full.
            Once full, returns:
            (exercise, confidence)
        """

        frame = np.array(
            [features.get(name, 0.0) for name in FEATURES],
            dtype=np.float32,
        )

        self.buffer.append(frame)

        if len(self.buffer) < SEQUENCE_LENGTH:
            return None

        sequence = np.array(
            self.buffer,
            dtype=np.float32,
        )

        # Normalize exactly like during training.
        sequence = self.scaler.transform(sequence)

        # Add batch dimension:
        # (60, 61) -> (1, 60, 61)
        sequence = np.expand_dims(sequence, axis=0)

        probabilities = self.model.predict(
            sequence,
            verbose=0,
        )[0]

        prediction = np.argmax(probabilities)

        exercise = self.encoder.inverse_transform(
            [prediction]
        )[0]

        confidence = float(probabilities[prediction])

        return exercise, confidence

    def clear(self):
        self.buffer.clear()