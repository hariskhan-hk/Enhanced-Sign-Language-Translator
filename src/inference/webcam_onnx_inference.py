# webcam.py
import cv2
import numpy as np
import onnxruntime
import time
from super_gradients.training.utils.detection_utils import DetectionVisualization

# CLASS NAMES (from your data.yaml)
CLASS_NAMES = ['C1', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C2', 'C20', 'C21', 'C22',
               'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C3', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35',
               'C36', 'C37', 'C38', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']



def preprocess_frame(frame, input_size=(640, 640)):
    """Preprocesses a single frame for ONNX model input."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
    frame = cv2.resize(frame, input_size)
    # IMPORTANT: Keep the data type as uint8! Don't normalize.
    # frame = frame.astype(np.float32) / 255.0  # REMOVE THIS LINE
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    return frame.astype(np.uint8) #Add the type casting here!

def run_inference(session, image_np):
    """Runs inference with the ONNX model."""
    inputs = [o.name for o in session.get_inputs()]
    outputs = [o.name for o in session.get_outputs()]
    result = session.run(outputs, {inputs[0]: image_np})
    return result


def draw_predictions(image, predictions, class_names):
    """Draws bounding boxes and labels on the image."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
    [flat_predictions] = predictions

    color_mapping = DetectionVisualization._generate_color_mapping(len(class_names))

    for (sample_index, x1, y1, x2, y2, class_score, class_index) in flat_predictions[flat_predictions[:, 0] == 0]:
        class_index = int(class_index)
        if class_index < len(class_names):  # Bounds check!
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
    scale_x = original_size[1] / 640
    scale_y = original_size[0] / 640

    [flat_predictions] = predictions
    flat_predictions[:, 1] *= scale_x  # x_min
    flat_predictions[:, 2] *= scale_y  # y_min
    flat_predictions[:, 3] *= scale_x  # x_max
    flat_predictions[:, 4] *= scale_y  # y_max

    return [flat_predictions]

def main():
    onnx_path = "sign_lang_model_best.onnx"  # Your exported ONNX model
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    session = onnxruntime.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        original_size = frame.shape[:2]
        input_frame = preprocess_frame(frame)
        start_time = time.time()
        predictions = run_inference(session, input_frame)
        predictions = post_process_predictions(predictions, original_size)
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        result_frame = draw_predictions(frame, predictions, CLASS_NAMES)
        cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Webcam Inference", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()