import streamlit as st
import cv2
import time
import pandas as pd
from PIL import Image
import mediapipe as mp
import base64

from pose_detection import create_landmarker, draw_pose_landmarks, POSE_CONNECTIONS, pose_result_store
from form_coach import FormCoach
from coach_ui import draw_coach_overlay

def get_landmark_name(idx):
    names = {
        11: "left shoulder", 12: "right shoulder", 13: "left elbow", 14: "right elbow",
        15: "left wrist", 16: "right wrist", 17: "left pinky", 18: "right pinky",
        19: "left index", 20: "right index", 21: "left thumb", 22: "right thumb",
        23: "left hip", 24: "right hip", 25: "left knee", 26: "right knee",
        27: "left ankle", 28: "right ankle", 29: "left heel", 30: "right heel",
        31: "left foot index", 32: "right foot index"
    }
    return names.get(idx, f"landmark_{idx}")

st.set_page_config(
    page_title="Exercise Coach",
    layout="wide"
)

if 'running' not in st.session_state:
    st.session_state.running = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'landmarker' not in st.session_state:
    with st.spinner("Loading pose landmarker model..."):
        st.session_state.landmarker = create_landmarker()
        st.success("Model loaded!")
if 'show_all_state' not in st.session_state:
    st.session_state.show_all_state = False


if 'selected_exercise' not in st.session_state:
    st.session_state.selected_exercise = "pushup"
if 'coach' not in st.session_state:
    st.session_state.coach = FormCoach(
        exercise=st.session_state.selected_exercise,
        model_dir="models"
    )
if 'coach_result' not in st.session_state:
    st.session_state.coach_result = None

# Title
st.title("Real-Time Pose Detection")

st.session_state.running = True
video_width = 700

col_ex1, col_ex2, col_reset = st.columns([1, 1, 1])
with col_ex1:
    if st.button("Squat"):
        st.session_state.selected_exercise = "squat"
        st.session_state.coach.switch_exercise("squat")
with col_ex2:
    if st.button("Push-up"):
        st.session_state.selected_exercise = "pushup"
        st.session_state.coach.switch_exercise("pushup")
with col_reset:
    if st.button("Reset Reps"):
        st.session_state.coach.reset_reps()

st.caption(f"Active exercise: **{st.session_state.selected_exercise.replace('_', ' ').title()}**")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Camera Feed")
    
    # Video placeholder
    video_placeholder = st.empty()
    
    # Controls below video
    col_screenshot, col_stop, col_clear = st.columns(3)
    with col_screenshot:
        screenshot_btn = st.button("Take Screenshot")
    with col_stop:
        if st.button("Stop Camera"):
            st.session_state.running = False
            st.rerun()
    with col_clear:
        if st.button("Clear Data"):
            st.session_state.frame_count = 0

with col2:
    st.subheader("Detected Landmarks")
    
    # Tabs for different views
    tab1, tab2 = st.tabs(["Coach", "Key Points"])
    
    with tab1:
        coach_placeholder = st.empty()

    with tab2:
        keypoints_placeholder = st.empty()

# Initialize camera
if st.session_state.running:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    prev_time = time.time()
    frame_timestamp_ms = 0
    
    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            inference_frame = cv2.resize(frame, (640, 360))
            rgb_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            current_timestamp = int(time.time() * 1000)
            if current_timestamp <= frame_timestamp_ms:
                current_timestamp = frame_timestamp_ms + 1

            st.session_state.landmarker.detect_async(mp_image, current_timestamp)
            results = pose_result_store.get()
            frame_timestamp_ms = current_timestamp
            
            display_frame = frame.copy()
            
            landmark_data = []
            if results and results.pose_landmarks:
                landmarks = results.pose_landmarks[0]
                
                display_frame = draw_pose_landmarks(display_frame, landmarks)

                coach_result = st.session_state.coach.update(landmarks)
                st.session_state.coach_result = coach_result
                cr = st.session_state.coach_result
                if cr:
                    colour_hex = "#{:02x}{:02x}{:02x}".format(*cr["colour"][::-1])  # BGR → RGB → hex
                    coach_placeholder.markdown(f"""
                        <div style="padding:16px;border-radius:10px;border-left:5px solid {colour_hex};background:rgba(0,0,0,0.05)">
                            <div style="font-size:13px;color:gray;margin-bottom:4px">{cr['exercise'].replace('_',' ').title()}</div>
                            <div style="font-size:28px;font-weight:600">{cr['reps']} reps</div>
                            <div style="font-size:15px;margin-top:8px">{cr['advice']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
                for idx in key_joints:
                    if idx < len(landmarks):
                        lm = landmarks[idx]
                        if lm.visibility > 0.5:
                            landmark_data.append({
                                "Landmark": idx,
                                "Name": get_landmark_name(idx),
                                "X": f"{lm.x:.3f}",
                                "Y": f"{lm.y:.3f}",
                                "Z": f"{lm.z:.3f}",
                                "Visibility": f"{lm.visibility:.2f}"
                            })
            
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            _, buf = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buf).decode()
            video_placeholder.markdown(f'<img src="data:image/jpeg;base64,{b64}" width="{video_width}">',
                                        unsafe_allow_html=True)
            
            st.session_state.frame_count += 1
            
            if landmark_data:
                keypoints_placeholder.dataframe(
                    pd.DataFrame(landmark_data),
                    width='stretch'
                )
            else:
                keypoints_placeholder.info("No landmarks detected")
            
            if results and results.pose_landmarks and st.session_state.show_all_state:
                all_data = []
                for idx, lm in enumerate(results.pose_landmarks[0]):
                    all_data.append({
                        "Index": idx,
                        "X": f"{lm.x:.3f}",
                        "Y": f"{lm.y:.3f}",
                        "Z": f"{lm.z:.3f}",
                        "Visibility": f"{lm.visibility:.2f}"
                    })
            
            if screenshot_btn:
                screenshot = Image.fromarray(display_frame_rgb)
                filename = f"screenshot_{int(time.time())}.png"
                screenshot.save(filename)
                st.success(f"Screenshot saved as {filename}!")
            
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())
    
    finally:
        cap.release()
else:
    video_placeholder.info("Click 'Start Camera' in the sidebar to begin")
    keypoints_placeholder.info("Landmark data will appear here")

# Footer
st.markdown("---")
st.markdown("Built by Abhinav Gupta")