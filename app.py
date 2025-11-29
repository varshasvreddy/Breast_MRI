import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from hybrid_model import HybridModel  # make sure hybrid_model.py is in same folder

# --------------------- App Header ---------------------
st.set_page_config(page_title="Breast MRI Detection", layout="centered")
st.title("🩺 Breast MRI Cancer Detection")
st.write("Upload a Breast MRI image to detect **Benign (Normal)** or **Malignant**.")

# --------------------- Model Loading ---------------------
@st.cache_resource
def load_model():
    model_path = "breast_mri_model.pth"  # your trained model path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HybridModel(num_classes=2)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
    except FileNotFoundError:
        st.error(f"Model file not found at '{model_path}'. Make sure it is uploaded.")
        return None, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

    return model, device

model, device = load_model()

# Stop if model not loaded
if model is None:
    st.stop()

# --------------------- Image Transform ---------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --------------------- File Upload ---------------------
uploaded_file = st.file_uploader("Upload a Breast MRI image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    try:
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

    except Exception as e:
        st.error(f"Error processing image: {e}")
