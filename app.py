import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
from huggingface_hub import hf_hub_download

# --- Model Configuration ---
MODEL_REPO = "jahangi/brain_tumer"          
MODEL_FILENAME = "brain_tumor_model.h5"
MODEL_PATH = MODEL_FILENAME

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Hugging Face, please wait... (first time only)")
    with st.spinner("Downloading..."):
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            local_dir=".",            
            local_dir_use_symlinks=False
        )
    st.success("Model downloaded successfully!")

# Load the model
@st.cache_resource(show_spinner=False)  # Ek baar load hone ke baad cache mein rakhega
def load_brain_tumor_model():
    return load_model(MODEL_PATH)

model = load_brain_tumor_model()

# --- Streamlit UI ---
st.title("🧠 Brain Tumor Detection System")
st.write("Upload a Brain MRI image (JPG, PNG, JPEG) and the model will predict **Tumor** or **No Tumor**.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI Image", width=300)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0               # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    if st.button("🔍 Detect Tumor"):
        with st.spinner("Analyzing image..."):
            prediction = model.predict(img_array)[0][0]
        
        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
        confidence = round(confidence, 2)

        if prediction > 0.5:
            st.error("⚠️ **Result: Tumor Detected**")
            st.write(f"**Confidence: {confidence}%**")
        else:
            st.success("✅ **Result: No Tumor Detected**")
            st.write(f"**Confidence: {confidence}%**")

st.markdown("---")
st.caption("Model trained on Brain Tumor MRI Dataset | Deployed via Streamlit + Hugging Face")

