import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Ye naya best way — huggingface_hub se direct download
@st.cache_resource(show_spinner="Downloading model from Hugging Face (~120MB)...")
def download_model():
    from huggingface_hub import hf_hub_download
    
    model_path = hf_hub_download(
        repo_id="jahangi/brain_tumer",
        filename="brain_tumor_model.h5"
    )
    return model_path

# Model load karo
@st.cache_resource(show_spinner="Loading AI model...")
def load_brain_model():
    path = download_model()
    return load_model(path)

# Title
st.title("🧠 Brain Tumor Detection System (AI Powered)")
st.write("Upload a Brain MRI image → AI will predict **Tumor** or **No Tumor**")

# Model load
model = load_brain_model()
st.success("✅ Model loaded successfully!")

# Image upload
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI", width=300)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    if st.button("🔍 Detect Tumor", type="primary"):
        with st.spinner("Analyzing..."):
            prediction = model.predict(img_array)[0][0]

        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

        if prediction > 0.5:
            st.error("⚠️ **TUMOR DETECTED**")
        else:
            st.success("✅ **NO TUMOR DETECTED**")

        st.write(f"**Confidence: {confidence:.2f}%**")
        st.progress(confidence / 100)
