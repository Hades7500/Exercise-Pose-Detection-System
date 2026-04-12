"""
Step 1 — Extract MediaPipe Landmarks from Exercise Videos
==========================================================
Uses the NEW MediaPipe Tasks API (PoseLandmarker) matching pose_detection.py.
Processes bicep curl, squat, and pushup videos from the Kaggle dataset,
extracts landmarks + joint angles per frame, saves to landmarks.csv.

Usage:
    python extract_landmarks.py --dataset_path "path/to/your/dataset"
    python extract_landmarks.py --dataset_path "path/to/dataset" --model_path "../models/pose_landmarker_lite.task"

Requirements:
    pip install mediapipe opencv-python numpy pandas tqdm
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm

# ── New Tasks API (matches pose_detection.py) ─────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode
# ─────────────────────────────────────────────────────────────────────────────

EXERCISE_FOLDERS = {
    "bicep_curl": ["barbell biceps curl", "bicep curl", "bicep_curl", "biceps curl"],
    "squat":      ["squat", "squats", "barbell squat"],
    "pushup":     ["push-up", "pushup", "push_up", "push up"],
}

LM = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow":    13, "right_elbow":    14,
    "left_wrist":    15, "right_wrist":    16,
    "left_hip":      23, "right_hip":      24,
    "left_knee":     25, "right_knee":     26,
    "left_ankle":    27, "right_ankle":    28,
}


def calculate_angle(a, b, c):
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba, bc  = a - b, c - b
    cosine  = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return round(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))), 2)


def get_coords(landmarks, name):
    lm = landmarks[LM[name]]
    return (lm.x, lm.y, lm.z)


def extract_features(landmarks) -> dict:
    row = {}

    for name, idx in LM.items():
        lm = landmarks[idx]
        row[f"{name}_x"]   = round(lm.x, 5)
        row[f"{name}_y"]   = round(lm.y, 5)
        row[f"{name}_z"]   = round(lm.z, 5)
        row[f"{name}_vis"] = round(lm.visibility, 3)

    row["left_elbow_angle"]     = calculate_angle(get_coords(landmarks, "left_shoulder"),  get_coords(landmarks, "left_elbow"),    get_coords(landmarks, "left_wrist"))
    row["right_elbow_angle"]    = calculate_angle(get_coords(landmarks, "right_shoulder"), get_coords(landmarks, "right_elbow"),   get_coords(landmarks, "right_wrist"))
    row["left_shoulder_angle"]  = calculate_angle(get_coords(landmarks, "left_elbow"),    get_coords(landmarks, "left_shoulder"), get_coords(landmarks, "left_hip"))
    row["right_shoulder_angle"] = calculate_angle(get_coords(landmarks, "right_elbow"),   get_coords(landmarks, "right_shoulder"),get_coords(landmarks, "right_hip"))
    row["left_knee_angle"]      = calculate_angle(get_coords(landmarks, "left_hip"),      get_coords(landmarks, "left_knee"),     get_coords(landmarks, "left_ankle"))
    row["right_knee_angle"]     = calculate_angle(get_coords(landmarks, "right_hip"),     get_coords(landmarks, "right_knee"),    get_coords(landmarks, "right_ankle"))
    row["left_hip_angle"]       = calculate_angle(get_coords(landmarks, "left_shoulder"), get_coords(landmarks, "left_hip"),      get_coords(landmarks, "left_knee"))
    row["right_hip_angle"]      = calculate_angle(get_coords(landmarks, "right_shoulder"),get_coords(landmarks, "right_hip"),     get_coords(landmarks, "right_knee"))

    row["elbow_symmetry"]    = abs(row["left_elbow_angle"]    - row["right_elbow_angle"])
    row["knee_symmetry"]     = abs(row["left_knee_angle"]     - row["right_knee_angle"])
    row["shoulder_symmetry"] = abs(row["left_shoulder_angle"] - row["right_shoulder_angle"])
    row["hip_symmetry"]      = abs(row["left_hip_angle"]      - row["right_hip_angle"])

    ls = get_coords(landmarks, "left_shoulder");  rs = get_coords(landmarks, "right_shoulder")
    lh = get_coords(landmarks, "left_hip");       rh = get_coords(landmarks, "right_hip")
    torso_h = abs(((ls[1]+rs[1])/2) - ((lh[1]+rh[1])/2)) + 1e-8
    row["torso_lean"] = round((((ls[0]+rs[0])/2) - ((lh[0]+rh[0])/2)) / torso_h, 4)

    return row


def resolve_exercise_label(folder_name):
    folder_lower = folder_name.lower().strip()
    for label, aliases in EXERCISE_FOLDERS.items():
        if any(alias.lower() in folder_lower or folder_lower in alias.lower() for alias in aliases):
            return label
    return None


def apply_form_labels(df: pd.DataFrame) -> pd.DataFrame:
    labels = []
    for _, row in df.iterrows():
        ex, label = row["exercise"], "good_form"
        if ex == "bicep_curl":
            avg_elbow    = (row["left_elbow_angle"]    + row["right_elbow_angle"])    / 2
            avg_shoulder = (row["left_shoulder_angle"] + row["right_shoulder_angle"]) / 2
            if avg_elbow > 150:       label = "incomplete_rep"
            elif avg_shoulder > 40:   label = "elbow_flare"
        elif ex == "squat":
            avg_knee = (row["left_knee_angle"] + row["right_knee_angle"]) / 2
            if avg_knee > 160:        label = "incomplete_rep"
            elif row["knee_symmetry"] > 20: label = "knee_cave"
            elif avg_knee < 60:       label = "knees_over_toes"
        elif ex == "pushup":
            avg_hip = (row["left_hip_angle"] + row["right_hip_angle"]) / 2
            if avg_hip < 155:         label = "hips_piked"
            elif avg_hip > 175 and abs(row["torso_lean"]) > 0.15: label = "hips_sagging"
        labels.append(label)
    df["form_label"] = labels
    return df


def process_dataset(dataset_path: str,
                    model_path:   str = "../models/pose_landmarker_lite.task",
                    output_csv:   str = "landmarks.csv"):

    all_rows, video_count, skipped = [], 0, 0

    all_folders = [f for f in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, f))]

    print(f"\n📁 Found {len(all_folders)} folders in dataset.")
    matched_folders = []
    for folder in all_folders:
        label = resolve_exercise_label(folder)
        if label:
            matched_folders.append((folder, label))
            print(f"  ✅  '{folder}'  →  {label}")
        else:
            print(f"  ⏭️   '{folder}'  →  skipped")

    if not matched_folders:
        print("\n❌ No matching folders found. Check EXERCISE_FOLDERS aliases at the top.")
        return
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at '{model_path}'.")
        print("   Make sure pose_landmarker_lite.task is in your models/ folder.")
        return

    print(f"\n▶️  Processing {len(matched_folders)} folders...\n")

    # IMAGE mode: processes each frame independently — no timestamp needed,
    # works cleanly with frame-skipping during offline extraction.
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        for folder_name, exercise_label in matched_folders:
            folder_path = os.path.join(dataset_path, folder_name)
            video_files = [f for f in os.listdir(folder_path)
                           if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
            print(f"📹  {exercise_label} — {len(video_files)} videos")

            for video_file in tqdm(video_files, desc=f"  {exercise_label}", unit="video"):
                cap = cv2.VideoCapture(os.path.join(folder_path, video_file))
                if not cap.isOpened():
                    skipped += 1
                    continue

                fps          = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_idx    = 0
                sample_every = max(1, int(fps / 10))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    if frame_idx % sample_every != 0:
                        continue

                    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result   = landmarker.detect(mp_image)

                    if not result.pose_landmarks:
                        continue

                    row = extract_features(result.pose_landmarks[0])
                    row["exercise"]   = exercise_label
                    row["video_file"] = video_file
                    row["frame_idx"]  = frame_idx
                    all_rows.append(row)

                cap.release()
                video_count += 1

    if not all_rows:
        print("\n❌ No landmarks extracted. Check that your videos are valid.")
        return

    df = pd.DataFrame(all_rows)
    df = apply_form_labels(df)
    df.to_csv(output_csv, index=False)

    print(f"\n✅  Done!")
    print(f"   Videos processed : {video_count}")
    print(f"   Videos skipped   : {skipped}")
    print(f"   Total frames     : {len(df)}")
    print(f"   Saved to         : {output_csv}\n")
    print("📊  Label distribution:")
    print(df.groupby(["exercise", "form_label"]).size().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_path",   type=str, default="../models/pose_landmarker_lite.task")
    parser.add_argument("--output_csv",   type=str, default="landmarks.csv")
    args = parser.parse_args()
    process_dataset(args.dataset_path, args.model_path, args.output_csv)