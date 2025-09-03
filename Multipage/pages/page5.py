import streamlit as st

st.title("Low Ligth Image Enhancement with Zero-DCE")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image directly
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Show file details
    st.write("Filename:", uploaded_file.name)
    st.write("File type:", uploaded_file.type)
    st.write("File size:", uploaded_file.size, "bytes")
