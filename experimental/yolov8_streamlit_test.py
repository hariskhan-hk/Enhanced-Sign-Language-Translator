import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile # To handle potential model file uploads if needed

# --- Streamlit App Interface ---
# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Urdu Sign Language Detection", layout="wide")

# --- Configuration ---
MODEL_PATH = 'model_weights/best.pt' # <<<--- IMPORTANT: Set the correct path to your trained model file!
CONFIDENCE_THRESHOLD = 0.5 # Adjust as needed (0.0 - 1.0)
WEBCAM_DEVICE_INDEX = 0 # Usually 0 for the default webcam, change if you have multiple

# --- Load Model ---
@st.cache_resource # Cache the model loading for efficiency
def load_yolo_model(model_path):
    """Loads the YOLOv8 model."""
    try:
        model = YOLO(model_path)
        st.success(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.error(f"Please ensure the path '{model_path}' is correct and the model file exists.")
        st.stop() # Stop the app if model loading fails

model = load_yolo_model(MODEL_PATH)

# --- Get Class Names ---
# Assuming the model object has the class names stored after training
try:
    class_names = model.names
    if not class_names or not isinstance(class_names, dict) or len(class_names) != 38:
         st.warning("Could not automatically determine class names or count doesn't match 38. Using generic names.")
         # Fallback if model.names isn't populated as expected
    #      class_names = {i: f'Class_{i}' for i in range(38)}
    # st.write("Detected Classes:", class_names) # Show the classes being used
except AttributeError:
     st.error("Failed to get class names from the model. The 'names' attribute might be missing.")
     st.warning("Using generic class names.")
     class_names = {i: f'Class_{i}' for i in range(38)} # Fallback


# --- Streamlit App Interface ---
st.title("Real-Time Urdu Sign Language Alphabet Detection")
st.caption(f"Using YOLOv8n | Model: {MODEL_PATH.split('/')[-1]} | Confidence Threshold: {CONFIDENCE_THRESHOLD}")
st.write("Detected Classes:", class_names) # Display class names here
st.markdown("---")

# --- Real-time Detection Logic ---
run_detection = st.checkbox("Start Real-time Detection", value=True)
frame_placeholder = st.empty() # Placeholder to display the video frames

# Attempt to open the webcam
cap = cv2.VideoCapture(WEBCAM_DEVICE_INDEX)

if not cap.isOpened():
    st.error(f"Error: Could not open webcam (Device Index: {WEBCAM_DEVICE_INDEX}).")
    st.info("Make sure the webcam is connected and not used by another application. Try changing the WEBCAM_DEVICE_INDEX if needed.")
else:
    st.success(f"Webcam (Device Index: {WEBCAM_DEVICE_INDEX}) opened successfully.")
    while run_detection and cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.warning("Failed to read frame from webcam. Stream ended or webcam disconnected.")
            break

        # --- Perform Inference ---
        # The model call automatically handles preprocessing
        results = model(frame, stream=True, verbose=False, conf=CONFIDENCE_THRESHOLD) # Set verbose=False to reduce console output

        # --- Process and Draw Results ---
        for r in results:
            boxes = r.boxes  # Boxes object for bounding box outputs
            for box in boxes:
                # 1. Get Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box (top-left, bottom-right)

                # 2. Get Confidence Score
                conf = round(float(box.conf[0]), 2)

                # 3. Get Class ID and Name
                cls_id = int(box.cls[0])
                class_name = class_names.get(cls_id, f"Unknown Class {cls_id}") # Get name, fallback if ID not in dict

                # 4. Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box

                # 5. Create and Draw Label
                label = f"{class_name}: {conf}"
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y1_label = max(y1, label_size[1] + 10) # Ensure label is within frame bounds
                # Put filled rectangle behind text for better readability
                cv2.rectangle(frame, (x1, y1_label - label_size[1] - 5),
                              (x1 + label_size[0], y1_label + base_line - 5),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (x1, y1_label - 7), # Position text slightly above the bottom of the rectangle
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) # Black text


        # --- Display Frame ---
        # Convert frame from BGR (OpenCV default) to RGB (Streamlit expects)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Small delay to prevent overwhelming resources (optional)
        # You might not need this with Streamlit's handling
        # cv2.waitKey(1)

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows() # Ensure OpenCV windows are closed if any were opened (less relevant in Streamlit)
    if not run_detection:
        st.info("Detection stopped.")
    st.write("Webcam released.")


# --- Optional: Add Instructions or Info ---
st.sidebar.title("Instructions")
st.sidebar.info(
    "1. Ensure your webcam is connected.\n"
    "2. Check the 'Start Real-time Detection' box.\n"
    "3. Position your hand showing an Urdu sign language alphabet in front of the webcam.\n"
    "4. The detected alphabet and confidence score will be shown.\n"
    f"5. Model used: {MODEL_PATH}\n"
    f"6. Minimum confidence for detection: {CONFIDENCE_THRESHOLD}"
)
st.sidebar.title("About")
st.sidebar.info("This app uses a YOLOv8n model trained on static Urdu sign language alphabet images to perform real-time detection via webcam.")