import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from enhance_net import enhance_net_nopool, load_model

st.title(" Low Light Image Enhancement with Zero-DCE")

# -----------------------
# Helper function
# -----------------------
def enhance_image(uploaded_file, model, device):
    # Load and preprocess
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    img = Image.open(uploaded_file).convert("RGB")
    input_img = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_img)

        # Handle tuple output from Zero-DCE
        if isinstance(output, tuple):
            enhanced = output[0]
        else:
            enhanced = output

    # Convert tensors to numpy
    original_np = input_img.squeeze().permute(1, 2, 0).cpu().numpy()
    enhanced_np = enhanced.squeeze().permute(1, 2, 0).cpu().numpy()

    # Clip to [0,1] for valid display
    original_np = np.clip(original_np, 0, 1)
    enhanced_np = np.clip(enhanced_np, 0, 1)

    return original_np, enhanced_np

# -----------------------
# Streamlit UI
# -----------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model().to(device).eval()

        # Enhance
        original_np, enhanced_np = enhance_image(uploaded_file, model, device)

        # Show results side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(original_np, use_container_width=True)
        with col2:
            st.subheader("Enhanced")
            st.image(enhanced_np, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
