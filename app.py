import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import urllib.request

st.title("Brain Tumor Detection System (AI Powered)")
st.write("Upload a Brain MRI image and the model will predict Tumor or No Tumor.")

MODEL_PATH = "brain_tumor_model.h5"

# YE WAALA LINK USE KARO (SAHI WALA)
MODEL_URL = "https://huggingface.co/jahangi/brain_tumer/resolve/main/brain_tumor_model.h5"

# Download only if not exists
if not os.path.exists(MODEL_PATH):
    st.warning("Model downloading for the first time... (~120MB)")
    with st.spinner("Downloading model from Hugging Face... (20-60 seconds)"):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("Model downloaded successfully!")

# Load model with cache
@st.cache_resource
def load_brain_model():
    return load_model(MODEL_PATH)

model = load_brain_model()

# Upload image
uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    if st.button("Detect Tumor"):
        with st.spinner("Analyzing image..."):
            prediction = model.predict(img_array)[0][0]

        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

        if prediction > 0.5:
            st.error("⚠️ Tumor Detected")
            st.write(f"**Confidence: {confidence:.2f}%**")
        else:
            st.success("✅ No Tumor Detected")
            st.write(f"**Confidence: {confidence:.2f}%**")
        
        st.progress(confidence / 100)
