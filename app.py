import streamlit as st
import numpy as np
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

MODEL_PATH = "brain_tumor_model.h5"
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1B9vbxl1vg0bFhVA_5SReq9J_aocJ46BA"


# Download model if not present
if not tf.io.gfile.exists(MODEL_PATH):
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# Load model
model = load_model(MODEL_PATH)

st.title("Brain Tumor Detection System (AI Powered)")
st.write("Upload a Brain MRI image and the model will predict Tumor or No Tumor.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((150, 150))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.error("Tumor Detected")
    else:
        st.success("No Tumor Detected")

