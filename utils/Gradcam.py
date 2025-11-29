import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from PIL import Image

def generate_gradcam(model, image: Image.Image, target_layer, class_idx=None, device=None):
    """
    Generate a Grad-CAM heatmap for a given image and model.

    Args:
        model: Trained PyTorch model
        image: PIL Image to visualize
        target_layer: Layer to visualize (e.g., model.backbone.layer4[-1])
        class_idx: (Optional) class index to target for Grad-CAM
        device: torch.device ("cpu" or "cuda")

    Returns:
        visualization: Numpy array with heatmap overlay
    """
    model.eval()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    img_rgb = np.array(image)
    rgb_img = np.float32(img_rgb) / 255
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Grad-CAM setup
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())

    targets = None
    if class_idx is not None:
        targets = [ClassifierOutputTarget(class_idx)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return visualization