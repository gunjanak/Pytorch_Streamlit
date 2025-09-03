import streamlit as st
from PIL import Image

st.title("Zero-DCE")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Show basic info
    st.write("Image format:", image.format)
    st.write("Image size:", image.size)