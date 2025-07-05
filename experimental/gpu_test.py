import cv2
from super_gradients.training import models
from super_gradients.common.object_names import Models
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.get('yolo_nas_s', num_classes=38, checkpoint_path = 'model_weights/ckpt_best.pth')
model = model.to(device)


output = model.predict_webcam()
