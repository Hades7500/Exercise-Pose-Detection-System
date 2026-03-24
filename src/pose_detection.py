import cv2
import mediapipe as mp
import numpy as np


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (12, 14), (13, 15), (15, 21), (15, 17), (15, 19), (16, 22), (16, 20),
    (14, 16), (16, 18), (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 31), (28, 31)
]

def draw_pose_landmarks(image, landmarks, connections=POSE_CONNECTIONS):
    h, w = image.shape[:2]
    
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 8, (0, 255, 0), -1)
        cv2.circle(image, (x, y), 8, (0, 0, 0), 2)
    
    for start, end in connections:
        start_pos = (int(landmarks[start].x * w), int(landmarks[start].y * h))
        end_pos = (int(landmarks[end].x * w), int(landmarks[end].y * h))
        cv2.line(image, start_pos, end_pos, (255, 0, 0), 4)
    
    return image

def create_landmarker(model_path='../models/pose_landmarker_lite.task'):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.8,
        min_pose_presence_confidence=0.8,
        min_tracking_confidence=0.8)
    return PoseLandmarker.create_from_options(options)

def main():
    model_path = 'pose_landmarker_lite.task'

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.8,
        min_pose_presence_confidence=0.8,
        min_tracking_confidence=0.8)

    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps_time = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_timestamp_ms = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 33  # for 30fps
            
            if results.pose_landmarks:
                frame = draw_pose_landmarks(frame, results.pose_landmarks[0])
            
            fps_time += 1
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('MediaPipe Webcam Pose', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()