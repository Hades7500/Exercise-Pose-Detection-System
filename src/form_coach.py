"""
form_coach.py — Real-time form analysis engine
===============================================
Loads the trained .pkl models and, given live MediaPipe landmarks,
returns form advice, rep counts, and confidence scores.

Used by coach_ui.py and imported into app.py.
"""

import os
import pickle
import numpy as np
from collections import deque

LM = {
    "left_shoulder":    11,
    "right_shoulder":   12,
    "left_elbow":       13,
    "right_elbow":      14,
    "left_wrist":       15,
    "right_wrist":      16,
    "left_hip":         23,
    "right_hip":        24,
    "left_knee":        25,
    "right_knee":       26,
    "left_ankle":       27,
    "right_ankle":      28,
}

ADVICE = {
    # Squat
    "knee_cave":        "Push your knees out, don't let them cave in.",
    "knees_over_toes":  "Sit back more, don't let knees go too far forward.",
    # Pushup
    "hips_piked":       "Lower your hips, keep a straight line head to heel.",
    "hips_sagging":     "Engage your core, don't let your hips sag.",
}

ADVICE_COLOUR = {
    "good_form":        (0, 210, 90),    # green
    "elbow_flare":      (0, 165, 255),   # orange
    "incomplete_rep":   (0, 165, 255),
    "knee_cave":        (0, 100, 255),   # red-orange
    "knees_over_toes":  (0, 100, 255),
    "hips_piked":       (0, 100, 255),
    "hips_sagging":     (0, 100, 255),
}


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# Mirrors extract_landmarks.py exactly
# ─────────────────────────────────────────────

def _coords(landmarks, name):
    lm = landmarks[LM[name]]
    return (lm.x, lm.y, lm.z)


def _angle(a, b, c):
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def extract_features(landmarks) -> dict:
    """Extract the same features as extract_landmarks.py from live landmarks."""
    row = {}

    # Raw coordinates
    for name in LM:
        lm = landmarks[LM[name]]
        row[f"{name}_x"]   = lm.x
        row[f"{name}_y"]   = lm.y
        row[f"{name}_z"]   = lm.z
        row[f"{name}_vis"] = lm.visibility

    # Elbow angles (bicep curl)
    row["left_elbow_angle"]  = _angle(_coords(landmarks, "left_shoulder"),
                                       _coords(landmarks, "left_elbow"),
                                       _coords(landmarks, "left_wrist"))
    row["right_elbow_angle"] = _angle(_coords(landmarks, "right_shoulder"),
                                       _coords(landmarks, "right_elbow"),
                                       _coords(landmarks, "right_wrist"))

    # Shoulder angles (elbow flare)
    row["left_shoulder_angle"]  = _angle(_coords(landmarks, "left_elbow"),
                                          _coords(landmarks, "left_shoulder"),
                                          _coords(landmarks, "left_hip"))
    row["right_shoulder_angle"] = _angle(_coords(landmarks, "right_elbow"),
                                          _coords(landmarks, "right_shoulder"),
                                          _coords(landmarks, "right_hip"))

    # Knee angles (squat)
    row["left_knee_angle"]  = _angle(_coords(landmarks, "left_hip"),
                                      _coords(landmarks, "left_knee"),
                                      _coords(landmarks, "left_ankle"))
    row["right_knee_angle"] = _angle(_coords(landmarks, "right_hip"),
                                      _coords(landmarks, "right_knee"),
                                      _coords(landmarks, "right_ankle"))

    # Hip angles (squat depth / pushup alignment)
    row["left_hip_angle"]  = _angle(_coords(landmarks, "left_shoulder"),
                                     _coords(landmarks, "left_hip"),
                                     _coords(landmarks, "left_knee"))
    row["right_hip_angle"] = _angle(_coords(landmarks, "right_shoulder"),
                                     _coords(landmarks, "right_hip"),
                                     _coords(landmarks, "right_knee"))

    # Symmetry
    row["elbow_symmetry"]    = abs(row["left_elbow_angle"]    - row["right_elbow_angle"])
    row["knee_symmetry"]     = abs(row["left_knee_angle"]     - row["right_knee_angle"])
    row["shoulder_symmetry"] = abs(row["left_shoulder_angle"] - row["right_shoulder_angle"])
    row["hip_symmetry"]      = abs(row["left_hip_angle"]      - row["right_hip_angle"])

    # Torso lean
    ls = _coords(landmarks, "left_shoulder")
    rs = _coords(landmarks, "right_shoulder")
    lh = _coords(landmarks, "left_hip")
    rh = _coords(landmarks, "right_hip")
    mid_sx = (ls[0] + rs[0]) / 2
    mid_hx = (lh[0] + rh[0]) / 2
    mid_sy = (ls[1] + rs[1]) / 2
    mid_hy = (lh[1] + rh[1]) / 2
    torso_h = abs(mid_sy - mid_hy) + 1e-8
    row["torso_lean"] = (mid_sx - mid_hx) / torso_h

    return row


# ─────────────────────────────────────────────
# REP COUNTER
# State-machine using angle thresholds
# ─────────────────────────────────────────────

class RepCounter:
    """
    Counts reps for a given exercise using a 2-state machine:
      DOWN → UP → DOWN = 1 rep
    Thresholds are per-exercise.
    """

    THRESHOLDS = {
        "squat":      {"up": 90,  "down": 160, "angle": "avg_knee"},
        "pushup":     {"up": 80,  "down": 160, "angle": "avg_elbow"},
    }

    def __init__(self, exercise: str):
        self.exercise = exercise
        self.count    = 0
        self.state    = "down"  # "up" | "down"
        self.cfg      = self.THRESHOLDS.get(exercise, self.THRESHOLDS["pushup"])

    def update(self, features: dict) -> int:
        """Feed current frame features, returns current rep count."""
        angle_key = self.cfg["angle"]

        if angle_key == "avg_elbow":
            angle = (features.get("left_elbow_angle", 180) +
                     features.get("right_elbow_angle", 180)) / 2
        elif angle_key == "avg_knee":
            angle = (features.get("left_knee_angle", 180) +
                     features.get("right_knee_angle", 180)) / 2
        else:
            angle = 180

        if self.state == "down" and angle < self.cfg["up"]:
            self.state = "up"
        elif self.state == "up" and angle > self.cfg["down"]:
            self.state = "down"
            self.count += 1

        return self.count

    def reset(self):
        self.count = 0
        self.state = "down"


# ─────────────────────────────────────────────
# FORM COACH — main class
# ─────────────────────────────────────────────

class FormCoach:
    """
    Load trained models and provide real-time form coaching.

    Usage:
        coach = FormCoach(exercise="bicep_curl", model_dir="models")

        # Inside your camera loop:
        result = coach.update(landmarks)
        advice_text   = result["advice"]
        advice_colour = result["colour"]
        rep_count     = result["reps"]
        confidence    = result["confidence"]
    """

    def __init__(self,
                 exercise:   str  = "pushup",
                 model_dir:  str  = "models",
                 smooth_window: int = 10,
                 min_confidence: float = 0.45):
        """
        Args:
            exercise:       "squat" | "pushup"
            model_dir:      folder containing the .pkl files
            smooth_window:  number of frames to smooth predictions over
            min_confidence: minimum confidence to show a non-good-form label
        """
        self.exercise       = exercise
        self.smooth_window  = smooth_window
        self.min_confidence = min_confidence

        self._model    = None
        self._encoder  = None
        self._features = None

        self._prediction_buffer = deque(maxlen=smooth_window)
        self._rep_counter       = RepCounter(exercise)
        self._last_result       = self._default_result()

        self._load_model(model_dir, exercise)

    # ── Model loading ─────────────────────────

    def _load_model(self, model_dir: str, exercise: str):
        path = os.path.join(model_dir, f"{exercise}.pkl")
        if not os.path.exists(path):
            print(f"[FormCoach] ⚠️  Model not found at '{path}'.")
            print(f"[FormCoach]    Run train_model.py first.")
            return

        with open(path, "rb") as f:
            bundle = pickle.load(f)

        self._model    = bundle["model"]
        self._encoder  = bundle["encoder"]
        self._features = bundle["features"]
        print(f"[FormCoach] ✅  Loaded {exercise} model  "
              f"(cv={bundle.get('cv_mean', 0):.1%})")

    def switch_exercise(self, exercise: str, model_dir: str = "models"):
        """Switch to a different exercise mid-session."""
        self.exercise = exercise
        self._rep_counter = RepCounter(exercise)
        self._prediction_buffer.clear()
        self._load_model(model_dir, exercise)

    # ── Inference ─────────────────────────────

    def update(self, landmarks) -> dict:
        """
        Process one frame of landmarks.
        Returns a result dict with advice, colour, reps, confidence.
        """
        if self._model is None:
            return self._default_result()

        try:
            features = extract_features(landmarks)
        except Exception:
            return self._last_result

        # Update rep counter
        reps = self._rep_counter.update(features)

        # Build feature vector in the exact order the model expects
        x = np.array([[features.get(f, 0.0) for f in self._features]])

        # Predict + confidence
        proba     = self._model.predict_proba(x)[0]
        pred_idx  = int(np.argmax(proba))
        confidence = float(proba[pred_idx])
        label      = self._encoder.inverse_transform([pred_idx])[0]

        # If not confident enough, fall back to good_form
        if confidence < self.min_confidence and label != "good_form":
            label = "good_form"

        # Smooth: add to buffer, use majority vote
        self._prediction_buffer.append(label)
        smoothed_label = max(set(self._prediction_buffer),
                             key=list(self._prediction_buffer).count)

        result = {
            "label":       smoothed_label,
            "advice":      ADVICE.get(smoothed_label, ""),
            "colour":      ADVICE_COLOUR.get(smoothed_label, (255, 255, 255)),
            "reps":        reps,
            "confidence":  confidence,
            "exercise":    self.exercise,
            "features":    features,   # expose for debugging / UI
        }
        self._last_result = result
        return result

    def reset_reps(self):
        self._rep_counter.reset()

    @staticmethod
    def _default_result() -> dict:
        return {
            "label":      "good_form",
            "advice":     "Select an exercise to begin.",
            "colour":     (180, 180, 180),
            "reps":       0,
            "confidence": 0.0,
            "exercise":   "",
            "features":   {},
        }

    @property
    def rep_count(self) -> int:
        return self._rep_counter.count