import streamlit as st
import numpy as np
from super_gradients.training import models
from functools import lru_cache
import cv2
import torch
import onnxruntime # For ONNX model
import time

# --- Configuration ---

# Configure page settings (Optional - Streamlit uses main page config by default)
# st.set_page_config(page_title="Model Comparison") # You can set a specific title if needed

# CLASS NAMES (Must match the training classes)
CLASS_NAMES = ['C1', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C2', 'C20', 'C21', 'C22',
               'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C3', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35',
               'C36', 'C37', 'C38', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

YOLONAS_WEIGHTS_PATH = '../models/yolonas_ckpt_best.pth'  # Path to the YOLO-NAS model weights
ONNX_MODEL_PATH = '../model_weights/sign_lang_model_best.onnx'  # Path to the ONNX model

# --- Model Loading (Cached specific to this page) ---

# Note: Using st.cache_resource ensures the model stays loaded across reruns within this page
@st.cache_resource
def load_yolonas_model_comp(weights_path):
    """Loads the YOLO-NAS model for comparison."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.get('yolo_nas_s', num_classes=len(CLASS_NAMES), checkpoint_path=weights_path)
        model.to(device)
        model.eval()
        print("YOLO-NAS Model loaded successfully for comparison.")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO-NAS model from {weights_path}: {e}")
        print(f"Error loading YOLO-NAS model: {e}")
        return None

@st.cache_resource
def load_onnx_model_comp(onnx_path):
    """Loads the ONNX model for comparison."""
    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        st.success(f"ONNX using: {session.get_providers()}")
        return session
    except Exception as e:
        st.warning(f"Could not load ONNX with CUDA ({e}). Trying CPU only.")
        try:
            providers = ["CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(onnx_path, providers=providers)
            st.success(f"ONNX using: {session.get_providers()}")
            return session
        except Exception as e_cpu:
            st.error(f"Failed to load ONNX model on CPU: {e_cpu}")
            return None

# --- Helper Functions (Specific or adapted for comparison) ---

def preprocess_frame(frame, input_size=(640, 640)):
    """Preprocesses a single frame for ONNX model input."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, input_size)
    frame_transposed = np.transpose(frame_resized, (2, 0, 1))
    input_tensor = np.expand_dims(frame_transposed, axis=0)
    return input_tensor.astype(np.uint8) # Ensure correct dtype

def run_onnx_inference(session, image_np):
    """Runs inference with the ONNX model."""
    if session is None:
        st.error("ONNX session not loaded.")
        return None
    try:
        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        result = session.run(outputs, {inputs[0]: image_np})
        return result
    except Exception as e:
        st.error(f"ONNX inference failed: {e}")
        return None

# --- Main UI for Comparison Page ---
st.title("🚀 Model Speed Comparison")
st.subheader("YOLO-NAS (.pth) vs. ONNX")
st.markdown("""
This page compares the inference speed of the original YOLO-NAS PyTorch model (`.pth`)
against its optimized ONNX version using your live webcam feed.
- **<font color='green'>Green Boxes</font>:** YOLO-NAS (.pth) Detections
- **<font color='blue'>Blue Boxes</font>:** ONNX Detections
""", unsafe_allow_html=True)


# --- Load Models ---
yolonas_model = load_yolonas_model_comp(YOLONAS_WEIGHTS_PATH)
onnx_session = load_onnx_model_comp(ONNX_MODEL_PATH)

if not yolonas_model or not onnx_session:
    st.error("One or both models failed to load. Cannot run comparison. Check console logs.")
    st.stop()

# --- Comparison Controls ---
st.markdown("---")
confidence_threshold = st.slider("Set confidence threshold for comparison", 0.0, 1.0, 0.5, 0.05, key="comp_conf_slider")

# Display GPU status
st.markdown("---")
st.header("Device Status")
if torch.cuda.is_available():
    st.success(f"GPU detected: {torch.cuda.get_device_name(0)}")
    onnx_providers = onnx_session.get_providers()
    if "CUDAExecutionProvider" in onnx_providers:
        st.info("ONNX is configured to use CUDA.")
    else:
        st.warning("ONNX is using CPU despite GPU availability (check ONNX Runtime build/drivers).")
else:
    st.warning("No GPU detected. Using CPU for both models.")


# --- Webcam Comparison Logic ---
st.markdown("---")
run_webcam = st.checkbox("Activate Webcam Comparison", key="webcam_comp_checkbox")
frame_placeholder = st.empty()
stop_button_pressed = st.button("Stop Comparison", key="webcam_comp_stop")

if run_webcam and not stop_button_pressed:
    webcam_index = 0
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        st.error(f"Could not open webcam (Device Index: {webcam_index}).")
    else:
        st.info(f"Webcam comparison active (Device Index: {webcam_index}). Press 'Stop Comparison' to end.")
        # Unique key for stop state
        stop_key_comp = "comparison_webcam_stopped"
        st.session_state[stop_key_comp] = False

        while cap.isOpened() and not st.session_state.get(stop_key_comp, False):
            if stop_button_pressed: # Check button *before* reading frame
                 st.session_state[stop_key_comp] = True
                 break

            ret, frame = cap.read()
            if not ret:
                st.warning("Could not read frame from webcam.")
                break

            comp_frame = frame.copy() # Keep original frame for drawing
            pth_frame_input = frame.copy() # Copy for pth model if it modifies input

            # --- YOLO-NAS (.pth) Inference ---
            pth_inference_time = 0.0
            pth_labels = []
            pth_confidences = []
            pth_bboxes = []
            try:
                frame_rgb_pth = cv2.cvtColor(pth_frame_input, cv2.COLOR_BGR2RGB)
                start_time = time.time()
                with torch.no_grad():
                     device = next(yolonas_model.parameters()).device # Get current device
                     # This predict expects RGB numpy uint8
                     pth_prediction = yolonas_model.predict(frame_rgb_pth, conf=confidence_threshold)
                pth_inference_time = time.time() - start_time

                if hasattr(pth_prediction, 'prediction'):
                    pred_data = pth_prediction.prediction
                    pth_labels = pred_data.labels.astype(int)
                    pth_confidences = pred_data.confidence
                    pth_bboxes = pred_data.bboxes_xyxy
            except Exception as e:
                st.error(f"YOLO-NAS (.pth) inference error: {e}")
                pth_inference_time = -1 # Indicate error

            # --- ONNX Inference ---
            onnx_inference_time = 0.0
            onnx_labels = []
            onnx_confidences = []
            onnx_bboxes = []
            try:
                input_onnx = preprocess_frame(frame) # Use the helper
                start_time = time.time()
                onnx_prediction = run_onnx_inference(onnx_session, input_onnx)
                onnx_inference_time = time.time() - start_time

                if onnx_prediction is not None and len(onnx_prediction) > 0 and len(onnx_prediction[0]) > 0:
                    # Assuming output shape [batch_size, num_predictions, 7]
                    # where each prediction is [image_idx, x1, y1, x2, y2, confidence, class_id]
                    flat_preds = onnx_prediction[0]
                    valid_preds = flat_preds[flat_preds[:, 5] >= confidence_threshold] # Filter by confidence here
                    onnx_bboxes = valid_preds[:, 1:5]
                    onnx_labels = valid_preds[:, 6].astype(int)
                    onnx_confidences = valid_preds[:, 5]

            except Exception as e:
                st.error(f"ONNX inference error: {e}")
                onnx_inference_time = -1 # Indicate error


            # --- Draw Bounding Boxes ---
            # Draw .pth results (Green)
            for bbox, label, confidence in zip(pth_bboxes, pth_labels, pth_confidences):
                if 0 <= label < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[label]
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(comp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"P: {class_name}: {confidence:.2f}" # Add 'P' for Pytorch
                    cv2.putText(comp_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw ONNX results (Blue)
            for bbox, label, confidence in zip(onnx_bboxes, onnx_labels, onnx_confidences):
                 if 0 <= label < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[label]
                    x1, y1, x2, y2 = map(int, bbox)
                    # Scale ONNX boxes back to original frame size if preprocessed
                    # Assuming preprocess_frame resized to 640x640
                    orig_h, orig_w = frame.shape[:2]
                    proc_h, proc_w = 640, 640 # Input size used in preprocess_frame
                    x1 = int(x1 * orig_w / proc_w)
                    y1 = int(y1 * orig_h / proc_h)
                    x2 = int(x2 * orig_w / proc_w)
                    y2 = int(y2 * orig_h / proc_h)

                    cv2.rectangle(comp_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label_text = f"O: {class_name}: {confidence:.2f}" # Add 'O' for ONNX
                    # Offset ONNX label slightly to avoid overlap
                    cv2.putText(comp_frame, label_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


            # --- Draw Inference Times ---
            pth_time_str = f"{pth_inference_time*1000:.1f} ms" if pth_inference_time >= 0 else "Error"
            onnx_time_str = f"{onnx_inference_time*1000:.1f} ms" if onnx_inference_time >= 0 else "Error"
            cv2.putText(comp_frame, f"YOLO-NAS (.pth): {pth_time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(comp_frame, f"ONNX: {onnx_time_str}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # --- Display Frame ---
            frame_rgb_display = cv2.cvtColor(comp_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb_display, channels="RGB", use_column_width=True)

            # Check stop button state inside the loop
            if st.session_state.get("webcam_comp_stop"):
                st.session_state[stop_key_comp] = True
                break

        cap.release()
        st.info("Webcam comparison stopped.")
        if stop_key_comp in st.session_state:
            del st.session_state[stop_key_comp]
        # Reset button state
        if "webcam_comp_stop" in st.session_state:
            del st.session_state["webcam_comp_stop"]


elif stop_button_pressed:
    st.info("Webcam comparison stopped.")
    if 'cap' in locals() and 'cap' in vars() and cap.isOpened():
        cap.release()
     # Reset button state
    if "webcam_comp_stop" in st.session_state:
        del st.session_state["webcam_comp_stop"]