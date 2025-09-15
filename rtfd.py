import cv2
from deepface import DeepFace
import time
import threading

# --- Configuration ---
# Use 'mtcnn' for better face detection
detector_backend = 'mtcnn'
# Time in seconds between each DeepFace analysis
analysis_interval = 0.5 

# --- Shared Variables ---
analysis_result = []
analysis_thread = None
is_analyzing = False

# --- Analysis Function (Runs in a separate thread) ---
def analyze_faces_thread(frame_copy):
    global analysis_result, is_analyzing
    is_analyzing = True
    try:
        results = DeepFace.analyze(
            frame_copy,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend=detector_backend
        )
        if results is not None and len(results) > 0:
            analysis_result = results
        else:
            analysis_result = [] # Clear results if no face is detected
    except Exception as e:
        print(f"DeepFace Analysis Error: {e}")
        analysis_result = []
    finally:
        is_analyzing = False

# --- Main Program ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print(f"Using '{detector_backend}' for face detection.")
print(f"Analyzing every {analysis_interval} seconds.")
print("Press 'q' to quit.")

last_analysis_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from video stream.")
        break
    
    current_time = time.time()
    
    # Check if a new analysis should start and if a previous one is not in progress
    if not is_analyzing and (current_time - last_analysis_time >= analysis_interval):
        # Create a deep copy of the frame to avoid issues with it changing
        # while being analyzed in the separate thread.
        frame_copy = frame.copy()
        analysis_thread = threading.Thread(target=analyze_faces_thread, args=(frame_copy,))
        analysis_thread.daemon = True # Allows the thread to exit with the main program
        analysis_thread.start()
        last_analysis_time = current_time

    # --- Drawing the results ---
    if analysis_result:
        for face_info in analysis_result:
            if 'region' in face_info:
                # Extract face bounding box coordinates
                x, y, w, h = face_info['region']['x'], face_info['region']['y'], \
                             face_info['region']['w'], face_info['region']['h']
                
                # Extract dominant emotion
                dominant_emotion = face_info['dominant_emotion']
                
                # Draw a green rectangle and text
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    dominant_emotion,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
    else:
        # If no result, indicate that analysis is in progress or no face is found
        status_text = "Analyzing..." if is_analyzing else "No face detected"
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the processed frame
    cv2.imshow('Real-time Emotion Detection', frame)
    
    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()