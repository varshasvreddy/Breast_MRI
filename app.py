import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from models.hybrid_model import HybridModel

# --------------------- App Header ---------------------
st.set_page_config(page_title="Breast MRI Detection", layout="centered")
st.title("🩺 Breast MRI Cancer Detection")
st.write("Upload a Breast MRI image to detect **Benign (Normal)** or **Malignant** and visualize Grad-CAM heatmap.")

# --------------------- Model Loading ---------------------
@st.cache_resource
def load_model():
    model_path = "models/breast_mri_model.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridModel(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# --------------------- Transform ---------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

# --------------------- File Upload ---------------------
uploaded_file = st.file_uploader("Upload a Breast MRI image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocess
    x = transform(img).unsqueeze(0).to(device)

    # --------------------- Prediction ---------------------
    with torch.no_grad():
        preds = model(x)
        probs = F.softmax(preds, dim=1)
        cls = preds.argmax(1).item()
        conf = probs[0][cls].item() * 100

    label = "🟢 Benign (Normal)" if cls == 0 else "🔴 Malignant"
    st.subheader("Prediction:")
    st.write(f"{label} — Confidence: **{conf:.2f}%**")

    # --------------------- Grad-CAM Visualization ---------------------
    st.write("### 🔍 Grad-CAM Visualization")
    img_cv = np.array(img)
    rgb_img = np.float32(img_cv) / 255

    # Grad-CAM setup
    target_layer = model.backbone.layer4[-1]  # last conv block
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    targets = [ClassifierOutputTarget(cls)]

    grayscale_cam = cam(input_tensor=x, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    st.image(visualization, caption="Grad-CAM Heatmap", use_container_width=True)
