# webcam.py (This remains largely the same, but we'll make a few key adjustments)
import cv2
import numpy as np
import onnxruntime
import time
from super_gradients.training.utils.detection_utils import DetectionVisualization

CLASS_NAMES = ['C1', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C2', 'C20', 'C21', 'C22',
'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C3', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35',
'C36', 'C37', 'C38', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

def preprocess_frame(frame, input_size=(640, 640)):
    """Preprocesses a single frame for ONNX model input."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
    frame = cv2.resize(frame, input_size)
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    return frame.astype(np.uint8)

def run_inference(session, image_np):
    """Runs inference with the ONNX model."""
    inputs = [o.name for o in session.get_inputs()]
    outputs = [o.name for o in session.get_outputs()]
    result = session.run(outputs, {inputs[0]: image_np})
    return result

def draw_predictions(image, predictions, class_names):
    """Draws only the highest confidence prediction with its label."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    [flat_predictions] = predictions
    
    # Filter predictions for the current image (sample_index == 0)
    image_predictions = flat_predictions[flat_predictions[:, 0] == 0]
    
    # Check if we have any predictions
    if len(image_predictions) == 0:
        cv2.putText(image, "No predictions", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return image
    
    # Find the prediction with highest confidence
    highest_conf_idx = np.argmax(image_predictions[:, 5])  # Column 5 is class_score
    prediction = image_predictions[highest_conf_idx]
    
    # Extract the prediction details
    sample_index, x1, y1, x2, y2, class_score, class_index = prediction
    class_index = int(class_index)
    
    color_mapping = DetectionVisualization._generate_color_mapping(len(class_names))
    
    if class_index < len(class_names):
        # Draw the single prediction with highest confidence
        image = DetectionVisualization.draw_box_title(
            image_np=image,
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
            class_id=class_index,
            class_names=class_names,
            color_mapping=color_mapping,
            box_thickness=3,  # Thicker box for visibility
            pred_conf=class_score,
        )
        
        # Add prominent display of the prediction at the top of the screen
        prediction_text = f"Predicted: {class_names[class_index]} ({class_score:.2f})"
        cv2.putText(image, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        print(f"Warning: class_index {class_index} out of range for class_names")
        cv2.putText(image, "Invalid prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return image


def post_process_predictions(predictions, original_size):
    """Scales bounding box coordinates back to the original image size."""
    scale_x = original_size[1] / 640
    scale_y = original_size[0] / 640

    [flat_predictions] = predictions
    flat_predictions[:, 1] *= scale_x  # x_min
    flat_predictions[:, 2] *= scale_y  # y_min
    flat_predictions[:, 3] *= scale_x  # x_max
    flat_predictions[:, 4] *= scale_y  # y_max

    return [flat_predictions]



# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import onnxruntime
import time
from super_gradients.training.utils.detection_utils import DetectionVisualization
from PIL import Image
import tempfile
import os
import imageio
from functools import lru_cache

# --- Configuration and Setup ---
CLASS_NAMES = ['C1', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C2', 'C20', 'C21', 'C22',
'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C3', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35',
'C36', 'C37', 'C38', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

st.set_page_config(
    page_title="Sign Language Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ONNX Model Loading (Cached) ---
@st.cache_resource
def load_onnx_model(onnx_path="sign_lang_model_best.onnx"):
    """Loads the ONNX model with caching."""
    session = onnxruntime.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return session

# --- Helper Functions (from webcam.py, adapted for Streamlit) ---

# These functions are identical to the ones in webcam.py, so we can reuse them.
# Just make sure they are defined *before* being called.  Good practice!
def preprocess_frame(frame, input_size=(640, 640)):
    """Preprocesses a single frame for ONNX model input."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
    frame = cv2.resize(frame, input_size)
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    return frame.astype(np.uint8)

def run_inference(session, image_np):
    """Runs inference with the ONNX model."""
    inputs = [o.name for o in session.get_inputs()]
    outputs = [o.name for o in session.get_outputs()]
    result = session.run(outputs, {inputs[0]: image_np})
    return result

def post_process_predictions(predictions, original_size):
    """Scales bounding box coordinates back to the original image size."""
    scale_x = original_size[1] / 640
    scale_y = original_size[0] / 640
    [flat_predictions] = predictions
    flat_predictions[:, 1] *= scale_x
    flat_predictions[:, 2] *= scale_y
    flat_predictions[:, 3] *= scale_x
    flat_predictions[:, 4] *= scale_y
    return [flat_predictions]

def draw_predictions(image, predictions, class_names):
    """Draws only the highest confidence prediction with its label."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    [flat_predictions] = predictions
    
    # Filter predictions for the current image (sample_index == 0)
    image_predictions = flat_predictions[flat_predictions[:, 0] == 0]
    
    # Check if we have any predictions
    if len(image_predictions) == 0:
        cv2.putText(image, "No predictions", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return image
    
    # Find the prediction with highest confidence
    highest_conf_idx = np.argmax(image_predictions[:, 5])  # Column 5 is class_score
    prediction = image_predictions[highest_conf_idx]
    
    # Extract the prediction details
    sample_index, x1, y1, x2, y2, class_score, class_index = prediction
    class_index = int(class_index)
    
    color_mapping = DetectionVisualization._generate_color_mapping(len(class_names))
    
    if class_index < len(class_names):
        # Draw the single prediction with highest confidence
        image = DetectionVisualization.draw_box_title(
            image_np=image,
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
            class_id=class_index,
            class_names=class_names,
            color_mapping=color_mapping,
            box_thickness=3,  # Thicker box for visibility
            pred_conf=class_score,
        )
        
        # Add prominent display of the prediction at the top of the screen
        prediction_text = f"Predicted: {class_names[class_index]} ({class_score:.2f})"
        cv2.putText(image, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        print(f"Warning: class_index {class_index} out of range for class_names")
        cv2.putText(image, "Invalid prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return image



# --- Streamlit App ---

# Load the ONNX model (this will be cached)
if 'onnx_session' not in st.session_state:
    st.session_state.onnx_session = load_onnx_model()

# --- Main UI ---

st.title("Sign Language Detection")

# --- Sidebar ---

with st.sidebar:
    st.image("images/logo.png", use_column_width=True)  # Replace with your logo
    st.markdown(
        """
        <div style='text-align: center; font-size: 20px; font-weight: bold;'>
        Catch every gesture,<br>Translate with ease!
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_data
def load_sample_images() -> dict:
    return {
        "sample1": "images/sample_img1.png",
        "sample2": "images/sample_img2.png",
        "sample3": "images/sample_img3.png"
    }

st.subheader("Get started by exploring the app")
sample_images = load_sample_images()
cols = st.columns(3)
for idx, (col, (_, img_path)) in enumerate(zip(cols, sample_images.items())):
    with col:
        st.image(img_path, caption=f"Sample {idx+1}")

# Features description
st.markdown("""
What can you do?

Detect sign language gestures from images, videos, or live webcam.

Real-time gesture detection with YOLO-NAS technology.

Upload and analyze sign language videos with automatic gesture recognition.
""")
# Detection interface

st.write("Ready to try? Click below to start detecting!")
if "detection_started" not in st.session_state:
    st.session_state.detection_started = False

if st.button("Start Detection"):
    st.session_state.detection_started = True

def process_video_batch(frames: list[np.ndarray], batch_size: int = 16) -> list[np.ndarray]:
    processed_frames = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:min(i + batch_size, len(frames))]
        batch_results = []  # Accumulate results for this batch
        for frame in batch:
            original_size = frame.shape[:2]
            input_frame = preprocess_frame(frame)
            predictions = run_inference(st.session_state.onnx_session, input_frame)
            predictions = post_process_predictions(predictions, original_size)
            result_frame = draw_predictions(frame, predictions, CLASS_NAMES)
            batch_results.append(result_frame)  # Append the processed frame
        processed_frames.extend(batch_results)  # Add results of this batch
    return processed_frames

if st.session_state.detection_started:
    st.header("Let's do some detection!")
    detection_type = st.selectbox(
        "Choose how you want to detect:",
        ("Choose an Option", "Image Upload", "Webcam", "Take a Picture")
    )

    if detection_type == "Webcam":
        if st.checkbox("Activate Webcam"):
            st.write("Webcam is active.")

            # Check for GPU
            gpu_info = st.empty()
            if "CUDAExecutionProvider" in st.session_state.onnx_session.get_providers():
                gpu_info.success("GPU detected and being used!")
            else:
                gpu_info.warning("No GPU detected. Using CPU.  Expect slower performance.")

            # Use st.image to display the video feed.  Key is important for updates!
            frame_placeholder = st.empty()
            # url = "http://192.168.100.11:8080/video"
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("Could not open webcam.")
            else:
                while True:  # Use a while loop for continuous frames
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Could not read frame.")
                        break

                    original_size = frame.shape[:2]
                    input_frame = preprocess_frame(frame)
                    start_time = time.time()  # For FPS calculation
                    predictions = run_inference(st.session_state.onnx_session, input_frame)
                    predictions = post_process_predictions(predictions, original_size)
                    end_time = time.time()
                    fps = 1 / (end_time - start_time)

                    result_frame = draw_predictions(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), predictions, CLASS_NAMES) #Keep consistant color format
                    result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                    cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    frame_placeholder.image(result_frame, channels="RGB")  # Display the frame

                    # Check for 'q' key press to exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()  # Important for cleanup!


    elif detection_type == "Image Upload":
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            # Save the uploaded image temporarily to use with OpenCV
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_image.getvalue())
                temp_img_path = tmp_file.name
            
            # Load image with OpenCV (same as in image_inference.py)
            image_np = cv2.imread(temp_img_path)
            if image_np is None:
                st.error("Failed to load image. Please try a different file.")
            else:
                # Display original image
                col1, col2 = st.columns(2)
                with col1:
                    display_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                    st.image(display_img, caption="Uploaded Image", use_column_width=True)
                
                with col2:
                    with st.spinner("Processing image..."):
                        # Get original size before any processing
                        original_size = image_np.shape[:2]
                        st.write(f"Original image dimensions: {original_size}")
                        
                        # Process with same flow as image_inference.py
                        input_frame = preprocess_frame(image_np)
                        
                        # Debug info
                        predictions = run_inference(st.session_state.onnx_session, input_frame)
                        st.write("Raw prediction shape:", [p.shape for p in predictions])
                        
                        # Post-process with explicit empty prediction check
                        predictions = post_process_predictions(predictions, original_size)
                        
                        # Use the original image for drawing (not the preprocessed one)
                        original_for_draw = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                        
                        # Draw predictions
                        result_image = draw_predictions(original_for_draw, predictions, CLASS_NAMES)
                        st.image(result_image, caption="Detected Image", use_column_width=True)
                        
                        # Process detected classes
                        flat_predictions = predictions[0]
                        if len(flat_predictions) > 0:
                            detected_labels = []
                            for pred in flat_predictions:
                                if len(pred) >= 7:  # Make sure we have enough elements
                                    cls_idx = int(pred[6])  # Class index is at position 6
                                    if 0 <= cls_idx < len(CLASS_NAMES):
                                        confidence = pred[5]  # Confidence score
                                        detected_labels.append(f"{CLASS_NAMES[cls_idx]} ({confidence:.2f})")
                            if detected_labels:
                                st.success(f"Detected: {', '.join(detected_labels)}")
                            else:
                                st.warning("No valid classes detected.")
                        else:
                            st.warning("No predictions found.")
                    
                # Clean up the temp file
                try:
                    os.unlink(temp_img_path)
                except:
                    pass


    elif detection_type == "Take a Picture":
        if picture := st.camera_input("Take a picture"):
            image = Image.open(picture)
            image_np = np.array(image)  # This is already in RGB format
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Captured Image", use_column_width=True)
            with col2:
                with st.spinner("Processing image..."):
                    # Get original size before any processing
                    original_size = image_np.shape[:2]
                    st.write(f"Original image dimensions: {original_size}")
                    
                    # Convert RGB to BGR for consistent processing with preprocess_frame
                    # The camera input is RGB but preprocess_frame expects BGR
                    image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    
                    # Process image with the BGR version
                    input_frame = preprocess_frame(image_np_bgr)
                    
                    # Debug info
                    predictions = run_inference(st.session_state.onnx_session, input_frame)
                    st.write("Raw prediction shape:", [p.shape for p in predictions])
                    
                    # Post-process predictions
                    predictions = post_process_predictions(predictions, original_size)
                    
                    # Draw predictions - use the BGR version for consistency
                    result_image = draw_predictions(image_np_bgr, predictions, CLASS_NAMES)
                    # Convert back to RGB for display in Streamlit
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    st.image(result_image, caption="Detected Image", use_column_width=True)
                    
                    # Process detected classes
                    flat_predictions = predictions[0]
                    if len(flat_predictions) > 0:
                        detected_labels = []
                        for pred in flat_predictions:
                            if len(pred) >= 7:  # Make sure we have enough elements
                                cls_idx = int(pred[6])  # Class index is at position 6
                                if 0 <= cls_idx < len(CLASS_NAMES):
                                    confidence = pred[5]  # Confidence score
                                    detected_labels.append(f"{CLASS_NAMES[cls_idx]} ({confidence:.2f})")
                        if detected_labels:
                            st.success(f"Detected: {', '.join(detected_labels)}")
                        else:
                            st.warning("No valid classes detected.")
                    else:
                        st.warning("No predictions found.")

st.markdown("""
Key Features:

🖼 Image Upload: Upload an image showing a sign language gesture.

📹 Video Upload: Analyze and translate sign language from video.

🎥 Live Webcam Detection: Detect gestures in real-time using your webcam.
""")

if st.button("Learn More"):
    st.header("About This App")
    cols = st.columns([1, 1])

    with cols[0]:
        st.write("""
            Sign Language is a powerful form of communication used by millions worldwide. 
            This app bridges the gap between spoken and sign languages using advanced AI models 
            for real-time gesture detection and translation.
        """)
    with cols[1]:
        st.write("""
            ### How It Works
            Using YOLO-NAS technology, our app detects and interprets hand gestures 
            representing letters and numbers in sign language. Upload media or use your 
            webcam for instant recognition and translation.
        """)

st.markdown("<br>", unsafe_allow_html=True)