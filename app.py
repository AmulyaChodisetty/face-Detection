import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from deepface import DeepFace
import av
import cv2
import threading
import time

# --- Streamlit UI Setup ---
st.title("Real-time Emotion Detection with DeepFace")
st.write("This app analyzes emotions from your webcam feed.")

# --- Shared State Variables for Analysis ---
# A global dictionary to hold the analysis results and control concurrency
shared_state = {"result": [], "lock": threading.Lock(), "is_analyzing": False}

# --- DeepFace Analysis Function ---
def analyze_deepface(frame):
    """
    Runs DeepFace analysis in a separate thread.
    """
    with shared_state["lock"]:
        shared_state["is_analyzing"] = True
    try:
        results = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='mtcnn'
        )
        with shared_state["lock"]:
            shared_state["result"] = results if results and len(results) > 0 else []
    except Exception as e:
        print(f"DeepFace Analysis Error: {e}")
        with shared_state["lock"]:
            shared_state["result"] = []
    finally:
        with shared_state["lock"]:
            shared_state["is_analyzing"] = False

# --- Video Transformer Class ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_analysis_time = time.time()
        self.analysis_interval = 0.5  # Time in seconds between analyses
        self.frame_copy = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        current_time = time.time()
        
        # Check if it's time for a new analysis and if a thread is not already running
        with shared_state["lock"]:
            if (current_time - self.last_analysis_time >= self.analysis_interval and
                not shared_state["is_analyzing"]):
                
                self.last_analysis_time = current_time
                # Create a deep copy to ensure the frame doesn't change during analysis
                self.frame_copy = img.copy() 
                
                # Start the analysis in a new thread
                analysis_thread = threading.Thread(target=analyze_deepface, args=(self.frame_copy,))
                analysis_thread.daemon = True
                analysis_thread.start()

        # Draw the last known results on the current frame
        with shared_state["lock"]:
            results = shared_state["result"]
            
            if results:
                for face_info in results:
                    if 'region' in face_info:
                        x, y, w, h = face_info['region']['x'], face_info['region']['y'], \
                                     face_info['region']['w'], face_info['region']['h']
                        dominant_emotion = face_info['dominant_emotion']
                        
                        # Draw bounding box and text
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            img,
                            dominant_emotion,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )
            else:
                # Provide real-time feedback
                status_text = "Analyzing..." if shared_state["is_analyzing"] else "No face detected"
                cv2.putText(img, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Return the processed frame to the web UI
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Run the Streamlit app ---
webrtc_streamer(key="emotion-detection", video_processor_factory=VideoProcessor)