import cv2
import numpy as np
import onnxruntime
import time
import torch
from super_gradients.training import models

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the .pth model
pth_model = models.get(
    'yolo_nas_s',
    num_classes=38,
    checkpoint_path='model_weights/ckpt_best.pth'
)
# Move model to GPU
pth_model.to(device)
pth_model.eval()  # Set to evaluation mode

# Load the ONNX model with GPU support
onnx_path = "sign_lang_model_best.onnx"
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)

# Check if CUDA is being used for ONNX
print(f"ONNX is using: {onnx_session.get_providers()}")

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

def preprocess_frame(frame, input_size=(640, 640)):
    """Preprocesses a single frame for ONNX model input."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
    frame = cv2.resize(frame, input_size)
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    return frame.astype(np.uint8)

def run_onnx_inference(session, image_np):
    """Runs inference with the ONNX model."""
    inputs = [o.name for o in session.get_inputs()]
    outputs = [o.name for o in session.get_outputs()]
    result = session.run(outputs, {inputs[0]: image_np})
    return result

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create a copy for PyTorch model (since predict modifies the input)
    pth_frame = frame.copy()
    
    # Perform prediction with .pth model
    with torch.no_grad():  # Disable gradient calculation for inference
        start_time = time.time()
        pth_prediction = pth_model.predict(pth_frame)
        pth_inference_time = time.time() - start_time

    # Perform prediction with ONNX model
    input_frame = preprocess_frame(frame)
    start_time = time.time()
    onnx_prediction = run_onnx_inference(onnx_session, input_frame)
    onnx_inference_time = time.time() - start_time

    # Extract prediction details for .pth model
    pth_bboxes = pth_prediction.prediction.bboxes_xyxy
    pth_labels = pth_prediction.prediction.labels.astype(int)
    pth_class_names = pth_prediction.class_names
    pth_confidences = pth_prediction.prediction.confidence

    # Extract prediction details for ONNX model
    onnx_bboxes = onnx_prediction[0][:, 1:5]
    onnx_labels = onnx_prediction[0][:, 6].astype(int)
    onnx_confidences = onnx_prediction[0][:, 5]

    # Iterate over detected objects for .pth model
    for bbox, label, confidence in zip(pth_bboxes, pth_labels, pth_confidences):
        class_name = pth_class_names[label]
        print(f"Detected {class_name} with confidence {confidence:.2f} using .pth model")

        # Draw bounding box and label on the frame
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Iterate over detected objects for ONNX model
    for bbox, label, confidence in zip(onnx_bboxes, onnx_labels, onnx_confidences):
        class_name = pth_class_names[label]
        print(f"Detected {class_name} with confidence {confidence:.2f} using ONNX model")

        # Draw bounding box and label on the frame
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame with annotations
    cv2.putText(frame, f".pth Inference Time: {pth_inference_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"ONNX Inference Time: {onnx_inference_time:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Webcam Inference', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()