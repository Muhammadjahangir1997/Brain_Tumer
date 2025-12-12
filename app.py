import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

st.title("Brain Tumor Detection System (AI Powered)")
st.write("Upload a Brain MRI image and the model will predict Tumor or No Tumor.")

MODEL_PATH = "brain_tumor_model.h5"

# Best solution: Use fuzzy=True
if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Downloading from Google Drive... (First time only)")
    with st.spinner("Downloading model (~150MB), please wait..."):
        try:
            gdown.download(
                "https://drive.google.com/file/d/1B9vbxl1vg0bFhVA_5SReq9J_aocJ46BA/view?usp=drive_link",
                MODEL_PATH,
                quiet=False,
                fuzzy=True                  # ← YE LINE ADD KAR DO
            )
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.info("Try uploading the model manually or use Hugging Face instead.")

# Load model
@st.cache_resource
def load_brain_model():
    return load_model(MODEL_PATH)

model = load_brain_model()

uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded MRI Image", width=300)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    if st.button("🔍 Detect Tumor"):
        with st.spinner("Analyzing..."):
            prediction = model.predict(img_array)[0][0]
        
        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
        
        if prediction > 0.5:
            st.error("⚠️ Tumor Detected")
        else:
            st.success("✅ No Tumor Detected")
            
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.progress(confidence / 100)


