import streamlit as st
from u2net import load_model,extract_foreground


st.title("Background Removal using U2-Net..")



# -----------------------
# Streamlit UI
# -----------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load model
        
        model = load_model()
        st.write("Model Loaded successfully")

        # # Enhance
        original_np, enhanced_np = extract_foreground(uploaded_file, model)

        # # Show results side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(original_np, use_container_width=True)
        with col2:
            st.subheader("Exracted Foreground")
            st.image(enhanced_np, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")