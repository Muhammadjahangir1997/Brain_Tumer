import streamlit as st
import numpy as np
import gdown
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.title("Brain Tumor Detection System (AI Powered)")
st.write("Upload a Brain MRI image and the model will predict Tumor or No Tumor.")

# Model storage
MODEL_PATH = "brain_tumor_model.h5"

# Direct Google Drive download link (file ID converted)
DRIVE_URL = "https://drive.google.com/uc?id=1B9vbxl1vg0bFhVA_5SReq9J_aocJ46BA"

# Download model if missing
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model, please wait..."):
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully.")

# Load the model
model = load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI Image", width=300)

    # Preprocessing
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Detect Tumor"):
        prediction = model.predict(img_array)[0][0]
        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

        if prediction > 0.5:
            st.success("Result: Tumor Detected")
        else:
            st.success("Result: No Tumor Detected")

        st.write(f"Confidence: {confidence:.2f}%")
