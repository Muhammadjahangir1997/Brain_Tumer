import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

# --- Model download (if not exists) ---
MODEL_PATH = "brain_tumor_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1B9vbxl1vg0bFhVA_5SReq9J_aocJ46BA"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model, please wait...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

# Load trained model
model = load_model(MODEL_PATH)

# --- Streamlit UI ---
st.title("Brain Tumor Detection System (AI Powered)")
st.write("Upload a Brain MRI image and the model will predict Tumor or No Tumor")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI Image", width=300)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Detect Tumor"):
        prediction = model.predict(img_array)[0][0]
        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

        if prediction > 0.5:
            st.success("Result: Tumor Detected")
        else:
            st.success("Result: No Tumor Detected")

        st.write(f"Confidence: {confidence:.2f}%")
