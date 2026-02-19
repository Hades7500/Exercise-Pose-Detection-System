import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from PIL import Image
import threading
from queue import Queue
import av
import mediapipe as mp

# Import your existing pose detection code
from pose_detection import create_landmarker, draw_pose_landmarks, POSE_CONNECTIONS

def get_landmark_name(idx):
    names = {
        0: "nose", 1: "left eye (inner)", 2: "left eye", 3: "left eye (outer)",
        4: "right eye (inner)", 5: "right eye", 6: "right eye (outer)",
        7: "left ear", 8: "right ear", 9: "mouth (left)", 10: "mouth (right)",
        11: "left shoulder", 12: "right shoulder", 13: "left elbow", 14: "right elbow",
        15: "left wrist", 16: "right wrist", 17: "left pinky", 18: "right pinky",
        19: "left index", 20: "right index", 21: "left thumb", 22: "right thumb",
        23: "left hip", 24: "right hip", 25: "left knee", 26: "right knee",
        27: "left ankle", 28: "right ankle", 29: "left heel", 30: "right heel",
        31: "left foot index", 32: "right foot index"
    }
    return names.get(idx, f"landmark_{idx}")

# Page configuration
st.set_page_config(
    page_title="Exercise Coach",
    # page_icon="💪",
    layout="wide"
)

# Initialize session state
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

# Title
st.title("Real-Time Pose Detection")

# # Sidebar controls
# with st.sidebar:
#     st.header("Controls")
    
#     # Camera control
#     st.session_state.running = True
    
#     # Confidence threshold
#     confidence = st.slider("Detection Confidence", 0.5, 1.0, 0.8, 0.05)
    
#     # Video width
#     video_width = st.slider("Video Width", 400, 1200, 700)
    
#     # Stats
#     st.subheader("Statistics")
#     stats_placeholder = st.empty()
    
#     # Model info
#     st.subheader("Model Info")
#     st.info("Using: pose_landmarker_lite.task")

st.session_state.running = True
video_width = 700

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Camera Feed")
    
    # Video placeholder
    video_placeholder = st.empty()
    
    # Controls below video
    col_screenshot, col_stop, col_clear = st.columns(3)
    with col_screenshot:
        screenshot_btn = st.button("📸 Take Screenshot")
    with col_stop:
        if st.button("⏹️ Stop Camera"):
            st.session_state.running = False
            st.rerun()
    with col_clear:
        if st.button("🔄 Clear Data"):
            st.session_state.frame_count = 0

with col2:
    st.subheader("Detected Landmarks")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Key Points", "Raw Data", "Connections"])
    
    with tab1:
        keypoints_placeholder = st.empty()
    
    with tab2:
        raw_data_placeholder = st.empty()
        # MOVE CHECKBOX HERE - outside the loop
        show_all_landmarks = st.checkbox(
            "Show All 33 Landmarks", 
            value=st.session_state.show_all_state,
            key="show_all_main",
            on_change=lambda: setattr(st.session_state, 'show_all_state', 
                                     not st.session_state.show_all_state)
        )
    
    with tab3:
        connections_placeholder = st.dataframe(
            pd.DataFrame(POSE_CONNECTIONS, columns=["Start", "End"]),
            width = 'stretch'
        )

# Initialize camera
if st.session_state.running:
    # Open camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # FPS calculation
    prev_time = time.time()
    frame_timestamp_ms = 0
    
    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            current_timestamp = int(time.time() * 1000)  # Current time in milliseconds

            # Ensure timestamp is increasing (if same as last, add 1)
            if current_timestamp <= frame_timestamp_ms:
                current_timestamp = frame_timestamp_ms + 1

            results = st.session_state.landmarker.detect_for_video(mp_image, current_timestamp)
            frame_timestamp_ms = current_timestamp  # Store for next comparison
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Draw landmarks if detected
            landmark_data = []
            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]
                
                # Draw using your function
                display_frame = draw_pose_landmarks(display_frame, landmarks)
                
                # Extract data for display
                key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]  # Key joints
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
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # cv2.putText(display_frame, f'FPS: {int(fps)}', (10, 30), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert BGR to RGB for Streamlit
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Display video with specified width
            video_placeholder.image(display_frame_rgb, channels="RGB", width=video_width)
            
            # Update stats
            st.session_state.frame_count += 1
            # stats_placeholder.metric("FPS", f"{fps:.1f}")
            # stats_placeholder.metric("Frames Processed", st.session_state.frame_count)
            # if results.pose_landmarks:
            #     stats_placeholder.metric("Pose Detected", "✅ Yes")
            # else:
            #     stats_placeholder.metric("Pose Detected", "❌ No")
            
            # Display key points
            if landmark_data:
                keypoints_placeholder.dataframe(
                    pd.DataFrame(landmark_data),
                    width = 'stretch'
                )
            else:
                keypoints_placeholder.info("No landmarks detected")
            
            # Display all landmarks if checkbox is checked (using session state)
            if results.pose_landmarks and st.session_state.show_all_state:
                all_data = []
                for idx, lm in enumerate(results.pose_landmarks[0]):
                    all_data.append({
                        "Index": idx,
                        "X": f"{lm.x:.3f}",
                        "Y": f"{lm.y:.3f}",
                        "Z": f"{lm.z:.3f}",
                        "Visibility": f"{lm.visibility:.2f}"
                    })
                raw_data_placeholder.dataframe(
                    pd.DataFrame(all_data),
                    width = 'stretch'
                )
            else:
                raw_data_placeholder.empty()
            
            # Handle screenshot
            if screenshot_btn:
                screenshot = Image.fromarray(display_frame_rgb)
                filename = f"screenshot_{int(time.time())}.png"
                screenshot.save(filename)
                st.success(f"Screenshot saved as {filename}!")
            
            # Small delay to control CPU usage
            time.sleep(0.01)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())
    
    finally:
        cap.release()

else:
    # Placeholder when camera is off
    video_placeholder.info("👈 Click 'Start Camera' in the sidebar to begin")
    keypoints_placeholder.info("Landmark data will appear here")
    raw_data_placeholder.info("Raw data will appear here when 'Show All 33 Landmarks' is checked")

# Footer
st.markdown("---")
st.markdown("Built by Abhinav Gupta")