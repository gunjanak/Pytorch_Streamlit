# import streamlit as st
# import torch
# from enhance_net import enhance_net_nopool,load_model

# st.title("Low Ligth Image Enhancement with Zero-DCE")

# # File uploader
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the image directly
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

#     # Show file details
#     st.write("Filename:", uploaded_file.name)
#     st.write("File type:", uploaded_file.type)
#     st.write("File size:", uploaded_file.size, "bytes")
    
# try:
#     model = load_model()
#     print(model.eval())
#     st.write(model.eval())
# except Exception as e:
#     st.write(e)


import streamlit as st
import torch
import numpy as np
from enhance_net import enhance_net_nopool, load_model
from torchvision import transforms
from PIL import Image

st.title("✨ Low Light Image Enhancement with Zero-DCE")

# ----------------------------
# Function: preprocess + infer
# ----------------------------
def enhance_image(uploaded_file, model, device):
    # Load image with PIL
    image = Image.open(uploaded_file).convert("RGB")

    # Transform to tensor [0,1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # (H,W,C) → (C,H,W), scaled to [0,1]
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)  # add batch dim

    # Inference
    with torch.no_grad():
        enhanced = model(input_tensor)

    # Convert back to image
    enhanced = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H,W,C)
    enhanced = np.clip(enhanced, 0, 1)  # keep in [0,1]
    enhanced_img = (enhanced * 255).astype(np.uint8)

    return Image.fromarray(enhanced_img)

# ----------------------------
# Streamlit UI
# ----------------------------
# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Load model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model()
        model.to(device).eval()

        # Enhance and display
        enhanced_img = enhance_image(uploaded_file, model, device)
        st.image(enhanced_img, caption="Enhanced Image", use_column_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
