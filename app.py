import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os


@st.cache_resource(show_spinner="Loading Brain Tumor AI Model...")
def load_model():
    model_path = "model.h5"

    if not os.path.exists(model_path):
        gdown.download(
            "https://drive.google.com/file/d/1B9vbxl1vg0bFhVA_5SReq9J_aocJ46BA/view?usp=sharing",
            model_path,
            quiet=False
        )

    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

st.title("🧠 Brain Tumor Detection")
st.write("Upload MRI → Get instant result")

uploaded_file = st.file_uploader(
    "Upload Brain MRI",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", width=300)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Detect Tumor"):
        with st.spinner("Analyzing..."):
            pred = model.predict(img_array)[0][0]

        confidence = pred * 100 if pred > 0.5 else (1 - pred) * 100

        if pred > 0.5:
            st.error("TUMOR DETECTED")
        else:
            st.success("NO TUMOR")

        st.write(f"Confidence: {confidence:.2f}%")
        st.progress(confidence / 100)


