import streamlit as st
import cv2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import Models

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

# --- Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.get('yolo_nas_s', num_classes=38, checkpoint_path='model_weights/ckpt_best.pth')  # Or pretrained_weights="coco"
model.to(device)

# --- Streamlit App ---
st.title("Object Detection with YOLO-NAS and Streamlit")

# Use a placeholder for the image/video display
frame_placeholder = st.empty()
detections_placeholder = st.empty()  # Placeholder for detection text

# --- Webcam Capture and Processing ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Could not open webcam.")
    st.stop()  # Stop the app if the webcam can't be opened

# Use a "running" flag to control the loop
running = st.checkbox("Run Webcam", value=True)  # Add a checkbox
gpu_info = st.empty()
if torch.cuda.is_available():
    gpu_info.success(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    gpu_info.warning("No GPU detected. Using CPU.")


while running:
    ret, frame = cap.read()
    if not ret:
        st.error("Could not read frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictions = model.predict(frame_rgb, conf=0.6, iou=0.6)  # Predict on the frame

    prediction = predictions  # Single image prediction
    class_names = prediction.class_names
    labels = prediction.prediction.labels
    confidence = prediction.prediction.confidence
    bboxes = prediction.prediction.bboxes_xyxy

    detected_objects_text = ""  # Store detections as a string
    for i, (label, conf, bbox) in enumerate(zip(labels, confidence, bboxes)):
        predicted_class_name = class_names[int(label)]
        detected_objects_text += f"- {predicted_class_name} (Confidence: {conf:.2f})\n" #append

        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{predicted_class_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Display the frame with bounding boxes
    frame_placeholder.image(frame, channels="BGR")  # Display BGR for OpenCV

    # Display the detected objects as text
    if detected_objects_text:
        detections_placeholder.markdown(f"**Detected Objects:**\n\n{detected_objects_text}")
    else:
        detections_placeholder.text("No objects detected.")

    #removed: no longer needed because of the checkbox, cv2.waitKey(1)  # Add a small delay (important for Streamlit)

# Release the webcam when the loop stops
cap.release()
#cv2.destroyAllWindows() # streamlit handles this