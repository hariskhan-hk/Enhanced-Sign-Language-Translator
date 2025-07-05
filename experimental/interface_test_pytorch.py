import streamlit as st
import cv2
import numpy as np
from super_gradients.training import models
from super_gradients.training.utils.detection_utils import DetectionVisualization  # Import for drawing
from PIL import Image
import tempfile
import os
import imageio
from functools import lru_cache
import torch  # Import PyTorch

# --- Configuration and Setup ---
CLASS_NAMES = ['C1', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C2', 'C20', 'C21', 'C22',
               'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C3', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35',
               'C36', 'C37', 'C38', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

st.set_page_config(
    page_title="Sign Language Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Model Loading (Cached) ---
@st.cache_resource
def load_model():
    """Loads the original PyTorch model with caching."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.get('yolo_nas_s', num_classes=38, checkpoint_path='model_weights/ckpt_best.pth')
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# --- Helper Functions ---
def preprocess_frame_pytorch(frame):
    """Preprocesses a single frame for PyTorch model input (with normalization)."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
    frame = cv2.resize(frame, (640, 640)) #resize
    frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
    frame = np.transpose(frame, (2, 0, 1))  # HWC to CHW
    frame = np.expand_dims(frame, axis=0)    # Add batch dimension
    return torch.from_numpy(frame)           # Convert to PyTorch tensor



def draw_predictions_pytorch(image, predictions, class_names):
    """Draws bounding boxes and labels using DetectionVisualization."""
    # Ensure image is in BGR format for OpenCV compatibility.
    if image.shape[2] == 3: # If it has 3 channels, assume RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.shape[0] == 3: #Also check for (3, H, W) format
         image = np.transpose(image, (1, 2, 0))
         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    color_mapping = DetectionVisualization._generate_color_mapping(len(class_names))

    # predictions is a list with length = batch_size
    # predictions[0] accesses the first (and only, in this case) image's predictions.
    # .prediction accesses the prediction object.
    # .bboxes_xyxy contains the bounding boxes as [x1, y1, x2, y2]
    # .confidence contains the confidence scores.
    # .labels contains the class labels.
    # Iterate directly through the prediction components.

    for bbox, score, label in zip(predictions[0].prediction.bboxes_xyxy,
                                  predictions[0].prediction.confidence,
                                  predictions[0].prediction.labels):
        x1, y1, x2, y2 = map(int, bbox)
        label = int(label)
        score = float(score)

        if label < len(class_names):  # Bounds check!
            image = DetectionVisualization.draw_box_title(
                image_np=image,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                class_id=label,
                class_names=class_names,
                color_mapping=color_mapping,
                box_thickness=2,
                pred_conf=score,
            )
        else:
            print(f"Warning: label {label} out of range for class_names")

    return image

# --- Streamlit App ---

# Load the model
if 'model' not in st.session_state:
    st.session_state.model = load_model()

# --- Main UI ---

st.title("Sign Language Detection")

# --- Sidebar ---
with st.sidebar:
    st.image("images/logo.png", use_column_width=True)
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
        "sample1": "images/alif.png",
        "sample2": "images/c38.jpg",
        "sample3": "images/alif_var.png"
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
    device = next(st.session_state.model.parameters()).device  # Get the device of the model

    for i in range(0, len(frames), batch_size):
        batch = frames[i:min(i + batch_size, len(frames))]
        batch_tensors = [preprocess_frame_pytorch(frame) for frame in batch]
        batch_tensor = torch.cat(batch_tensors).to(device)

        with torch.no_grad():  # Disable gradient calculation
            predictions = st.session_state.model(batch_tensor)

        batch_results = []
        for j, prediction in enumerate(predictions):
          # Draw predictions for each image
          drawn_frame = draw_predictions_pytorch(batch[j], [prediction], CLASS_NAMES)
          batch_results.append(drawn_frame)

        processed_frames.extend(batch_results)
    return processed_frames


if st.session_state.detection_started:
    st.header("Let's do some detection!")
    detection_type = st.selectbox(
        "Choose how you want to detect:",
        ("Choose an Option", "Image Upload", "Video Upload", "Webcam", "Take a Picture")
    )

    if detection_type == "Webcam":
        if st.checkbox("Activate Webcam"):
            st.write("Webcam is active.")

            # GPU Check
            gpu_info = st.empty()
            if torch.cuda.is_available():
                gpu_info.success("GPU detected and being used!")
            else:
                gpu_info.warning("No GPU detected.  Using CPU. Expect slower performance.")

            frame_placeholder = st.empty()
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("Could not open webcam.")
            else:
                device = next(st.session_state.model.parameters()).device  # Get device
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Could not read frame.")
                        break

                    # Preprocess and run inference
                    input_tensor = preprocess_frame_pytorch(frame).to(device)
                    with torch.no_grad():  # Disable gradient calculation
                        predictions = st.session_state.model(input_tensor)

                    # Draw predictions and display.
                    # Pass predictions as a list.
                    result_frame = draw_predictions_pytorch(frame, [predictions], CLASS_NAMES)
                    result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)  # For display
                    frame_placeholder.image(result_frame, channels="RGB")

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()

    elif detection_type == "Image Upload":
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            image_np = np.array(image)  # Convert PIL Image to NumPy array

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            with col2:
                with st.spinner("Processing image..."):
                    device = next(st.session_state.model.parameters()).device
                    input_tensor = preprocess_frame_pytorch(image_np).to(device)
                    with torch.no_grad():
                        predictions = st.session_state.model(input_tensor)

                    result_image = draw_predictions_pytorch(image_np, [predictions], CLASS_NAMES)
                    st.image(result_image, caption="Detected Image", use_column_width=True)

                    # Extract and display detected labels
                    detected_labels = [CLASS_NAMES[int(l)] for l in predictions.prediction.labels]
                    st.write("Detected Class Labels:", list(set(detected_labels)))


    elif detection_type == "Video Upload":
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        if uploaded_video:
            with st.spinner("Processing video..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    temp_file.write(uploaded_video.read())
                    input_path = temp_file.name

                output_path = os.path.join('Video', 'output_video.mp4')
                os.makedirs('Video', exist_ok=True)

                frames = []
                cap = cv2.VideoCapture(input_path)
                fps = cap.get(cv2.CAP_PROP_FPS)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()

                processed_frames = process_video_batch(frames)  # Corrected function call

                with imageio.get_writer(output_path, fps=fps) as writer:
                    for frame in processed_frames:
                        writer.append_data(frame)

                col1, col2 = st.columns(2)
                with col1:
                    st.video(input_path)
                with col2:
                    st.video(output_path)

                # Cleanup
                try:
                    os.remove(input_path)
                    os.remove(output_path)
                except OSError:
                    st.warning("Temporary files will be cleaned up later.")


    elif detection_type == "Take a Picture":
        if picture := st.camera_input("Take a picture"):
            image = Image.open(picture)
            image_np = np.array(image)  # Convert to NumPy array

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Captured Image", use_column_width=True)
            with col2:
                with st.spinner("Processing image..."):
                    device = next(st.session_state.model.parameters()).device
                    input_tensor = preprocess_frame_pytorch(image_np).to(device)
                    with torch.no_grad():
                        predictions = st.session_state.model(input_tensor)

                    result_image = draw_predictions_pytorch(image_np, [predictions], CLASS_NAMES)
                    st.image(result_image, caption="Detected Image", use_column_width=True)
                    detected_labels = [CLASS_NAMES[int(l)] for l in predictions.prediction.labels]
                    st.write("Detected Class Labels:", list(set(detected_labels)))

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