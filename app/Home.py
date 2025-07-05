import streamlit as st
import imageio
import numpy as np
from super_gradients.training import models
from super_gradients.training.utils.detection_utils import DetectionVisualization # For YOLO-NAS drawing
from PIL import Image
import tempfile
import os
from functools import lru_cache
import cv2
from typing import List, Tuple
import torch # Keep torch for device check
from ultralytics import YOLO # Import YOLO from ultralytics
import time # Keep for potential delays if needed

# --- Configuration ---

# Configure page settings
st.set_page_config(
    page_title="Enhanced Sign Language Detection",
    layout="wide",
    initial_sidebar_state="expanded",
    # You can add a favicon here if you have one:
    # page_icon="👋"
)

# CLASS NAMES (Ensure these match the classes BOTH models were trained on)
CLASS_NAMES = ['C1', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C2', 'C20', 'C21', 'C22',
               'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C3', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35',
               'C36', 'C37', 'C38', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

YOLONAS_WEIGHTS_PATH = '../models/yolonas_ckpt_best.pth' # Path to your YOLO-NAS weights
YOLOV8_WEIGHTS_PATH = '../models/yolov8_best.pt' # <<< CHANGE THIS to your trained YOLOv8 model path

# --- Model Loading (Cached) ---

@lru_cache(maxsize=1)
def load_yolonas_model(weights_path):
    """Loads the YOLO-NAS model."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.get('yolo_nas_s', num_classes=len(CLASS_NAMES), checkpoint_path=weights_path)
        model.to(device)
        model.eval() # Set to evaluation mode
        print("YOLO-NAS Model loaded successfully.")
        return model
    except FileNotFoundError:
        st.error(f"Error: YOLO-NAS weights file not found at {weights_path}. Please ensure the path is correct.")
        print(f"Error: YOLO-NAS weights file not found at {weights_path}.")
        return None
    except Exception as e:
        st.error(f"Error loading YOLO-NAS model: {e}")
        print(f"Error loading YOLO-NAS model: {e}")
        return None

@lru_cache(maxsize=1)
def load_yolov8_model(weights_path):
    """Loads the YOLOv8 model."""
    try:
        model = YOLO(weights_path)
        print(f"YOLOv8 Model loaded successfully from {weights_path}.")
        return model
    except FileNotFoundError:
        st.error(f"Error: YOLOv8 weights file not found at {weights_path}. Please ensure the path is correct.")
        print(f"Error: YOLOv8 weights file not found at {weights_path}.")
        return None
    except Exception as e:
        st.error(f"Error loading YOLOv8 model: {e}")
        print(f"Error loading YOLOv8 model: {e}")
        return None

# --- Helper Functions ---

@st.cache_data # Keep caching for static assets
def load_sample_images() -> dict:
    """Loads paths to sample images."""
    # Use placeholder images if samples don't exist, or handle error more gracefully
    base_image_path = "../data/sample_images"
    sample_files = {
        "sample1": os.path.join(base_image_path, "sample_img_1.png"),
        "sample2": os.path.join(base_image_path, "sample_img_2.png"),
        "sample3": os.path.join(base_image_path, "sample_img_3.png")
    }
    # Check if files exist
    existing_samples = {}
    for key, path in sample_files.items():
        if os.path.exists(path):
            existing_samples[key] = path
        else:
            st.warning(f"Sample image not found: {path}. Skipping.")
            print(f"Warning: Sample image not found: {path}")
    return existing_samples

def draw_predictions_yolonas(image_np: np.ndarray, predictions, class_names: list) -> np.ndarray:
    """Draws bounding boxes using SuperGradients DetectionVisualization."""
    if image_np.shape[2] == 3:
         image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
         image_bgr = image_np # Assuming it might already be BGR from cv2 read

    color_mapping = DetectionVisualization._generate_color_mapping(len(class_names))

    if hasattr(predictions, 'prediction'):
        bboxes = predictions.prediction.bboxes_xyxy
        scores = predictions.prediction.confidence
        labels = predictions.prediction.labels
        prediction_class_names = predictions.class_names # Use class names from prediction if available

        # Validate prediction_class_names (optional but good practice)
        if not prediction_class_names or len(prediction_class_names) != len(class_names):
            print("Warning: Class names mismatch or missing in YOLO-NAS prediction object. Using global CLASS_NAMES.")
            prediction_class_names_to_use = class_names
        else:
            prediction_class_names_to_use = prediction_class_names

        for bbox, score, label in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = map(int, bbox)
            label_idx = int(label)
            if 0 <= label_idx < len(prediction_class_names_to_use):
                image_bgr = DetectionVisualization.draw_box_title(
                    image_np=image_bgr,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    class_id=label_idx,
                    class_names=prediction_class_names_to_use, # Use potentially corrected names
                    color_mapping=color_mapping,
                    box_thickness=2,
                    pred_conf=score,
                )
            else:
                print(f"Warning: YOLO-NAS label index {label_idx} out of range for class names list (length {len(prediction_class_names_to_use)}).")
    else:
        print("Warning: Unexpected YOLO-NAS prediction format. Cannot draw boxes.")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def draw_predictions_yolov8(image_np: np.ndarray, results) -> Tuple[np.ndarray, List[str]]:
    """Draws bounding boxes using YOLOv8 results and returns annotated image + labels."""
    annotated_frame = image_np.copy() # Work on a copy
    detected_labels = []

    if results and len(results) > 0 and results[0].boxes:
        boxes = results[0].boxes
        class_name_map = results[0].names # Get class names from the model results

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = round(float(box.conf[0]), 2)
            cls_id = int(box.cls[0])
            class_name = class_name_map.get(cls_id, f"Unknown {cls_id}") # Use map safely

            label_text = f"{class_name} ({conf:.2f})"
            detected_labels.append(label_text)

            # Drawing settings
            color = (0, 255, 0) # BGR for OpenCV - Green
            thickness = 2
            font_scale = 0.6
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            # Draw filled background for text
            label_size, base_line = cv2.getTextSize(label_text, font, font_scale, thickness)
            y1_label = max(y1, label_size[1] + 10)
            cv2.rectangle(annotated_frame, (x1, y1_label - label_size[1] - 5),
                          (x1 + label_size[0], y1_label + base_line - 5),
                          color, cv2.FILLED)

            # Put text
            cv2.putText(annotated_frame, label_text, (x1, y1_label - 7),
                        font, font_scale, (0, 0, 0), thickness) # Black text

    return annotated_frame, detected_labels # Return both image and labels


# --- Initialize Session State ---
if 'yolonas_model' not in st.session_state:
    st.session_state.yolonas_model = None
if 'yolov8_model' not in st.session_state:
    st.session_state.yolov8_model = None
if 'selected_model_type' not in st.session_state:
    st.session_state.selected_model_type = "YOLO-NAS"
# Removed 'detection_started' as the detection UI shows directly now
# Removed: 'onnx_session' state

# --- Sidebar ---
with st.sidebar:
    # Use columns for better logo/tagline alignment if needed
    # col1, col2 = st.columns([1, 3])
    # with col1:
    #     st.image("images/logo.png", width=80) # Adjust width as needed
    # with col2:
    #     st.markdown("## Sign AI") # App name
    # Use full width if logo is wide
    logo_path = "static/images/logo_app.png"
    if os.path.exists(logo_path):
        st.image(logo_path, use_column_width=True)
    else:
        st.warning("Logo image 'images/logo.png' not found.")
        st.markdown("## Sign Language AI") # Fallback title

    st.markdown(
        """
        <div style='text-align: center; font-size: 18px; font-weight: bold; margin-top: -10px; margin-bottom: 10px;'>
            Catch every gesture,<br>Translate with ease!
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    st.header("⚙️ Configuration")

    # --- Model Selection ---
    st.subheader("Detection Model")
    model_type = st.radio(
        "Choose the detection model:",
        ("YOLO-NAS", "YOLOv8"),
        index=0 if st.session_state.selected_model_type == "YOLO-NAS" else 1,
        key="model_select_radio",
        # label_visibility="collapsed" # Hides the label "Choose the detection model:" if subheader is enough
    )
    # Update session state immediately when radio changes
    if st.session_state.selected_model_type != model_type:
        st.session_state.selected_model_type = model_type
        st.rerun() # Rerun to trigger potential model loading spinner immediately

    # --- Load Model ---
    # Trigger model loading based on selection
    model_loaded = False
    if st.session_state.selected_model_type == "YOLO-NAS":
        if st.session_state.yolonas_model is None:
            with st.spinner("Loading YOLO-NAS Model... Please wait."):
                st.session_state.yolonas_model = load_yolonas_model(YOLONAS_WEIGHTS_PATH)
        if st.session_state.yolonas_model is not None:
             model_loaded = True
             st.session_state.current_model = st.session_state.yolonas_model # Keep track of current active model
        else:
             st.error("YOLO-NAS model failed to load.")
    elif st.session_state.selected_model_type == "YOLOv8":
        if st.session_state.yolov8_model is None:
            with st.spinner("Loading YOLOv8 Model... Please wait."):
                st.session_state.yolov8_model = load_yolov8_model(YOLOV8_WEIGHTS_PATH)
        if st.session_state.yolov8_model is not None:
            model_loaded = True
            st.session_state.current_model = st.session_state.yolov8_model # Keep track of current active model
        else:
            st.error("YOLOv8 model failed to load.")

    # --- Confidence Threshold ---
    st.divider()
    st.subheader("Confidence Threshold")
    confidence_threshold = st.slider(
        "Minimum confidence to display detection:",
        0.0, 1.0, 0.5, 0.05,
        help="Adjust the sensitivity. Lower values detect more, potentially less accurate signs."
    )

    # --- Device Status ---
    st.divider()
    st.subheader("Device Status")
    if torch.cuda.is_available():
        st.success(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("⚠️ No GPU detected. Using CPU.")
        if st.session_state.selected_model_type == "YOLO-NAS":
             st.info("YOLO-NAS may be slower on CPU.")
        elif st.session_state.selected_model_type == "YOLOv8":
             st.info("YOLOv8 inference might be slower on CPU.")

# --- Main Content Area ---
st.title("👋 Welcome to the Sign Language Detection App!")
st.markdown("Interpret sign language gestures from images or your webcam in real-time.")

# --- Introductory Section ---
with st.container():
    st.header("✨ Explore Examples")
    st.write("See how the detection works with these sample images:")
    sample_images = load_sample_images()
    if sample_images:
        cols = st.columns(len(sample_images))
        for idx, (col, (key, img_path)) in enumerate(zip(cols, sample_images.items())):
            with col:
                try:
                    st.image(img_path, caption=f"Sample {idx+1}", use_column_width=True)
                except Exception as e:
                    st.warning(f"Could not load {key}: {e}")
    else:
        st.info("No sample images found in the 'images/' directory.")

    st.markdown("""
    **How to Use:**
    1.  Select a **Detection Model** and adjust the **Confidence Threshold** in the sidebar (⚙️).
    2.  Choose your input method below (Image Upload, Webcam, or Take Picture).
    3.  View the results!
    """)
    # Removed comparison link as it's not in this code
    # - Compare model speeds on the **Comparison** page (see sidebar).

st.divider()

# --- Detection Interface ---
st.header("👇 Choose Your Detection Method")

if not model_loaded:
    st.error(f"The selected **{st.session_state.selected_model_type}** model is not loaded. Please check the configuration and logs in the sidebar.")
    st.stop() # Stop execution if the selected model isn't ready

# Get the currently active model from session state
current_model = st.session_state.get('current_model', None)

detection_type = st.selectbox(
    "Select Input:",
    ("Choose an Option", "🖼️ Image Upload", "💻 Webcam", "📸 Take a Picture"),
    key="detection_type_select",
    # label_visibility="collapsed" # Hide label if header is enough
)

# --- Detection Logic Execution ---
if detection_type != "Choose an Option":

    # Use a container for the output area for better visual grouping
    results_container = st.container()
    # You can add a border or background for emphasis using markdown:
    # results_container.markdown("<div style='background-color: #f0f2f6; border-radius: 10px; padding: 15px; border: 1px solid #e0e0e0;'>", unsafe_allow_html=True)


    # --- Webcam Detection ---
    if detection_type == "💻 Webcam":
        with results_container:
            st.subheader("Live Webcam Detection")
            st.info("Click 'Activate Webcam' to start. Ensure you grant browser permissions.")

            col1, col2 = st.columns([1,1]) # Columns for buttons
            with col1:
                run_webcam = st.checkbox("Activate Webcam", key="webcam_checkbox")
            with col2:
                stop_button_pressed = st.button("Stop Webcam", key="webcam_stop", disabled=not run_webcam)

            frame_placeholder = st.empty()
            labels_placeholder = st.empty() # Placeholder for detected labels text

            if run_webcam and not stop_button_pressed:
                webcam_index = 0 # Or allow selection if multiple cams
                cap = cv2.VideoCapture(webcam_index)

                if not cap.isOpened():
                    st.error(f"❌ Error: Could not open webcam (Device Index: {webcam_index}). Try selecting a different index or check connections.")
                else:
                    st.success(f"✅ Webcam (Device Index: {webcam_index}) active. Point hands towards the camera. Press 'Stop Webcam' to end.")
                    # Use a unique key for the stop state
                    stop_key = "main_webcam_stopped"
                    st.session_state[stop_key] = False

                    while cap.isOpened() and not st.session_state.get(stop_key, False):
                        # Re-check button state inside the loop for immediate stopping
                        if st.session_state.get("webcam_stop"):
                            st.session_state[stop_key] = True
                            break

                        ret, frame = cap.read()
                        if not ret:
                            st.warning("⚠️ Could not read frame from webcam. Stream might have ended.")
                            break

                        # --- Perform Inference ---
                        result_image = frame # Default to original frame
                        detected_labels_str = "No detections"
                        frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert for display

                        try:
                            if st.session_state.selected_model_type == "YOLO-NAS":
                                # YOLO-NAS expects RGB
                                frame_rgb_predict = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                predictions = current_model.predict(frame_rgb_predict, conf=confidence_threshold)
                                result_image_rgb = draw_predictions_yolonas(frame_rgb_predict, predictions, CLASS_NAMES) # Draw takes RGB, returns RGB
                                frame_rgb_display = result_image_rgb # Update display frame

                                if hasattr(predictions, 'prediction') and predictions.prediction.labels is not None and len(predictions.prediction.labels) > 0:
                                    labels = predictions.prediction.labels
                                    scores = predictions.prediction.confidence
                                    detected_labels = [f"{CLASS_NAMES[int(l)]} ({s:.2f})" for l, s in zip(labels, scores) if 0 <= int(l) < len(CLASS_NAMES)]
                                    detected_labels_str = ", ".join(detected_labels) if detected_labels else "No detections above threshold"
                                else:
                                    detected_labels_str = "No detections"

                            elif st.session_state.selected_model_type == "YOLOv8":
                                # YOLOv8 from ultralytics can often handle BGR directly
                                results = current_model(frame, stream=False, verbose=False, conf=confidence_threshold) # Process frame
                                # Draw predictions using helper function
                                result_image_bgr, detected_labels = draw_predictions_yolov8(frame, results) # Draw takes BGR, returns BGR
                                frame_rgb_display = cv2.cvtColor(result_image_bgr, cv2.COLOR_BGR2RGB) # Convert final BGR to RGB for display

                                detected_labels_str = ", ".join(detected_labels) if detected_labels else "No detections above threshold"

                            # Update placeholders
                            frame_placeholder.image(frame_rgb_display, channels="RGB", use_column_width=True)
                            labels_placeholder.info(f"Detected: **{detected_labels_str}**")

                        except Exception as e:
                             st.error(f"❌ Error during webcam prediction: {e}")
                             frame_placeholder.image(frame_rgb_display, channels="RGB", use_column_width=True) # Show last good frame or original
                             labels_placeholder.error("Prediction failed.")
                             # Consider stopping the loop on error
                             # st.session_state[stop_key] = True
                             # break

                        # Give Streamlit a tiny break to update UI smoothly
                        # time.sleep(0.01)


                    cap.release()
                    if st.session_state.get(stop_key, False): # Check if stopped via button or end of loop
                        st.info("⏹️ Webcam stopped.")
                    if stop_key in st.session_state:
                        del st.session_state[stop_key]
                    # Reset button state after stop to allow restart without full refresh
                    if "webcam_stop" in st.session_state:
                        st.session_state["webcam_stop"] = False # Reset internal state if needed

            elif stop_button_pressed:
                 st.info("⏹️ Webcam stopped.")
                 # Ensure cleanup if stop is pressed right after activation but before loop runs
                 if 'cap' in locals() and 'cap' in vars() and cap.isOpened():
                     cap.release()
                 # Reset button state
                 st.session_state["webcam_stop"] = False # Reset internal state

    # --- Image Upload Detection ---
    elif detection_type == "🖼️ Image Upload":
        with results_container:
            st.subheader("Upload an Image")
            uploaded_image = st.file_uploader("Choose an image file (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])

            if uploaded_image:
                try:
                    image = Image.open(uploaded_image).convert("RGB")
                    image_np = np.array(image)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original Image**")
                        st.image(image_np, use_column_width=True)

                    with col2:
                        st.markdown("**Detection Results**")
                        result_placeholder = st.empty() # Placeholder for the result image
                        label_placeholder = st.empty() # Placeholder for the labels
                        result_placeholder.info("⏳ Processing... Please wait.")

                        with st.spinner("Running detection..."):
                            result_image = image_np
                            detected_labels_str = "No detections"

                            try:
                                if st.session_state.selected_model_type == "YOLO-NAS":
                                    predictions = current_model.predict(image_np, conf=confidence_threshold)
                                    result_image = draw_predictions_yolonas(image_np, predictions, CLASS_NAMES)
                                    if hasattr(predictions, 'prediction') and predictions.prediction.labels is not None and len(predictions.prediction.labels) > 0:
                                        labels = predictions.prediction.labels
                                        scores = predictions.prediction.confidence
                                        detected_labels = [f"{CLASS_NAMES[int(l)]} ({s:.2f})" for l, s in zip(labels, scores) if 0 <= int(l) < len(CLASS_NAMES)]
                                        detected_labels_str = ", ".join(detected_labels) if detected_labels else "No detections above threshold"
                                    else:
                                        detected_labels_str = "No detections"

                                elif st.session_state.selected_model_type == "YOLOv8":
                                    results = current_model.predict(image_np, conf=confidence_threshold, verbose=False)
                                    result_image_bgr, detected_labels = draw_predictions_yolov8(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), results) # Convert to BGR for drawing
                                    result_image = cv2.cvtColor(result_image_bgr, cv2.COLOR_BGR2RGB) # Convert back to RGB for display
                                    detected_labels_str = ", ".join(detected_labels) if detected_labels else "No detections above threshold"

                                result_placeholder.image(result_image, caption="Detected Image", use_column_width=True)
                                label_placeholder.success(f"Detected: **{detected_labels_str}**")

                            except Exception as e:
                                st.error(f"❌ Error during image prediction: {e}")
                                result_placeholder.image(image_np, caption="Processing Error", use_column_width=True) # Show original on error
                                label_placeholder.error("Prediction failed.")

                except Exception as e:
                    st.error(f"❌ Error loading image file: {e}")

    # --- Take a Picture Detection ---
    elif detection_type == "📸 Take a Picture":
        with results_container:
            st.subheader("Use Camera Input")
            st.info("Allow browser access to your camera, then click 'Take Photo'.")
            picture = st.camera_input("Take a photo using your webcam:")

            if picture:
                try:
                    image = Image.open(picture).convert("RGB")
                    image_np = np.array(image)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Captured Image**")
                        st.image(image_np, use_column_width=True)

                    with col2:
                        st.markdown("**Detection Results**")
                        result_placeholder = st.empty()
                        label_placeholder = st.empty()
                        result_placeholder.info("⏳ Processing... Please wait.")

                        with st.spinner("Running detection..."):
                            result_image = image_np
                            detected_labels_str = "No detections"
                            try:
                                if st.session_state.selected_model_type == "YOLO-NAS":
                                    predictions = current_model.predict(image_np, conf=confidence_threshold)
                                    result_image = draw_predictions_yolonas(image_np, predictions, CLASS_NAMES)
                                    if hasattr(predictions, 'prediction') and predictions.prediction.labels is not None and len(predictions.prediction.labels) > 0:
                                        labels = predictions.prediction.labels
                                        scores = predictions.prediction.confidence
                                        detected_labels = [f"{CLASS_NAMES[int(l)]} ({s:.2f})" for l, s in zip(labels, scores) if 0 <= int(l) < len(CLASS_NAMES)]
                                        detected_labels_str = ", ".join(detected_labels) if detected_labels else "No detections above threshold"
                                    else:
                                        detected_labels_str = "No detections"

                                elif st.session_state.selected_model_type == "YOLOv8":
                                    results = current_model.predict(image_np, conf=confidence_threshold, verbose=False)
                                    result_image_bgr, detected_labels = draw_predictions_yolov8(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), results) # Convert to BGR for drawing
                                    result_image = cv2.cvtColor(result_image_bgr, cv2.COLOR_BGR2RGB) # Convert back to RGB for display
                                    detected_labels_str = ", ".join(detected_labels) if detected_labels else "No detections above threshold"

                                result_placeholder.image(result_image, caption="Detected Image", use_column_width=True)
                                label_placeholder.success(f"Detected: **{detected_labels_str}**")

                            except Exception as e:
                                st.error(f"❌ Error during camera picture prediction: {e}")
                                result_placeholder.image(image_np, caption="Processing Error", use_column_width=True)
                                label_placeholder.error("Prediction failed.")

                except Exception as e:
                     st.error(f"❌ Error processing captured image: {e}")

    # Optional: Close the container div if you added one with markdown
    # results_container.markdown("</div>", unsafe_allow_html=True)

# --- Footer / About Section ---
st.divider()

# Using expander for less clutter, or keep as is if preferred
with st.expander("ℹ️ About This App & Features", expanded=False):
    st.markdown("""
    ### ✨ App Highlights:
    - **🖼️ Image Upload:** Analyze static images for sign language gestures.
    - **💻 Live Webcam:** Get real-time detection using your computer's camera.
    - **📸 Take a Picture:** Capture a photo directly within the app for analysis.
    - **🚀 Model Choice:** Switch between `YOLO-NAS` and `YOLOv8` models.
    - **🎚️ Confidence Control:** Adjust the detection sensitivity via the sidebar slider.
    """)
    # Removed comparison text

    st.markdown("---") # Inner divider

    st.header("Purpose")
    cols_about = st.columns([1, 1])
    with cols_about[0]:
        st.write("""
            Sign Language is a vital communication tool. This application leverages
            Artificial Intelligence to recognize and interpret common sign language gestures,
            aiming to bridge communication gaps.
        """)
    with cols_about[1]:
        st.write(f"""
            **How It Works:**
            Using the powerful **{st.session_state.selected_model_type}** object detection model,
            this app identifies hand shapes and positions corresponding to specific signs (currently trained for {CLASS_NAMES[0]} to {CLASS_NAMES[-1]}).
            Simply provide an image or use your webcam!
        """)

st.markdown(
    """
    <div style='text-align: center; margin-top: 20px; font-size: 12px; color: grey;'>
        Sign Language Detection App | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)