# image_inference.py
import cv2
import numpy as np
import onnxruntime
import os
import matplotlib.pyplot as plt
from super_gradients.training.utils.detection_utils import DetectionVisualization

# CLASS NAMES (from your data.yaml)
CLASS_NAMES = ['C1', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C2', 'C20', 'C21', 'C22',
               'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C3', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35',
               'C36', 'C37', 'C38', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']


def preprocess_image(image_path, input_size=(640, 640)):
    """Preprocesses the image for ONNX model input (uint8)."""
    image = cv2.imread(image_path)
    original_size = image.shape[:2]  # Get original size HERE, before resizing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
    image = cv2.resize(image, input_size)
    # Keep the data type as uint8.
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image.astype(np.uint8) , original_size # Return ONLY the processed image


def run_inference(onnx_path, image_np):
    """Runs inference with the ONNX model."""
    session = onnxruntime.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inputs = [o.name for o in session.get_inputs()]
    outputs = [o.name for o in session.get_outputs()]
    result = session.run(outputs, {inputs[0]: image_np})
    return result

def draw_predictions(image, predictions, class_names):
    """Draws bounding boxes and labels on the image."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    [flat_predictions] = predictions
    if flat_predictions.size == 0:  # Check if flat_predictions is empty
        print("No detections to draw.")
        return image

    color_mapping = DetectionVisualization._generate_color_mapping(len(class_names))

    for (sample_index, x1, y1, x2, y2, class_score, class_index) in flat_predictions[flat_predictions[:, 0] == 0]:
        class_index = int(class_index)
        if class_index < len(class_names):
            image = DetectionVisualization.draw_box_title(
                image_np=image,
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                class_id=class_index,
                class_names=class_names,
                color_mapping=color_mapping,
                box_thickness=2,
                pred_conf=class_score,
            )
        else:
            print(f"Warning: class_index {class_index} out of range for class_names")

    return image


def post_process_predictions(predictions, original_size):
    """Scales bounding box coordinates back to the original image size."""
    [flat_predictions] = predictions

    if flat_predictions.size == 0:
        return [flat_predictions]

    original_height, original_width = original_size
    print(f"Original Size: {original_size}")

    scale_x = original_width / 640.0
    scale_y = original_height / 640.0
    print(f"Scale X: {scale_x}, Scale Y: {scale_y}")

    print("Before Scaling:", flat_predictions)

    flat_predictions[:, 1] *= scale_x
    flat_predictions[:, 2] *= scale_y
    flat_predictions[:, 3] *= scale_x
    flat_predictions[:, 4] *= scale_y

    print("After Scaling:", flat_predictions)

    return [flat_predictions]

def main():
    # USE AN ABSOLUTE PATH HERE.
    image_path = "images/alif.png"  # REPLACE WITH YOUR ABSOLUTE IMAGE PATH
    image_path = os.path.abspath(image_path)
    onnx_path = "sign_lang_model_best.onnx"

    print(f"Attempting to load image from: {image_path}")

    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found at {image_path}")
        return

    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX model file not found at {onnx_path}")
        return

        # Preprocess the image
    input_image, original_size = preprocess_image(image_path)
    print("Original image size:", original_size)

    # Run inference
    predictions = run_inference(onnx_path, input_image)
    print("Raw Predictions (before post-processing):", predictions)

    predictions = post_process_predictions(predictions, original_size)
    print("Predictions (after post-processing):", predictions)


    # Load the original image for drawing (no need to reload, but for clarity)
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)


    # Draw predictions
    result_image = draw_predictions(original_image, predictions, CLASS_NAMES)

    # Save the result to a file
    cv2.imwrite("output_image.jpg", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    print("Image with detections (if any) saved to output_image.jpg")

    # Optional: Display with Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(result_image)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()