# export.py
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.conversion import DetectionOutputFormatMode
from super_gradients.conversion.conversion_enums import ExportQuantizationMode
from super_gradients.conversion import ExportTargetBackend

# DEFINE YOUR MODEL AND CHECKPOINT PATH HERE
num_classes = 38  # Correctly set based on your dataset
checkpoint_path = "model_weights/ckpt_best.pth"  #  Your model's checkpoint
onnx_filename = "sign_lang_model_best.onnx"

model = models.get(
    Models.YOLO_NAS_S,  # Matches your training script
    num_classes=num_classes,
    checkpoint_path=checkpoint_path,
)

export_result = model.export(
    onnx_filename,
    confidence_threshold=0.5, # Inference-time parameter
    nms_threshold=0.5,       # Inference-time parameter
    num_pre_nms_predictions=100, # Inference-time, usually fine
    max_predictions_per_image=50,  # Inference-time, usually fine
    output_predictions_format=DetectionOutputFormatMode.FLAT_FORMAT, # Good choice
    # engine=ExportTargetBackend.TENSORRT,  # Optional
)

print(f"ONNX model exported to: {onnx_filename}")
print(export_result)