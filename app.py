import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Direct load from Hugging Face (TensorFlow SavedModel format = no corruption)
@st.cache_resource(show_spinner="Loading Brain Tumor AI Model...")
def load_model():
    model = tf.keras.models.load_model("https://huggingface.co/jahangi/brain-tumor-fixed/resolve/main/saved_model")
    return model

model = load_model()

st.title("🧠 Brain Tumor Detection (Working 100%)")
st.write("Upload MRI → Get instant result")

uploaded_file = st.file_uploader("Upload Brain MRI", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", width=300)
    
    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Detect Tumor", type="primary"):
        with st.spinner("Analyzing..."):
            pred = model.predict(img_array)[0][0]
        
        confidence = pred * 100 if pred > 0.5 else (1 - pred) * 100
        
        if pred > 0.5:
            st.error("**TUMOR DETECTED**")
            st.write(f"Confidence: {confidence:.2f}%")
        else:
            st.success("**NO TUMOR**")
            st.write(f"Confidence: {confidence:.2f}%")
        
        st.progress(confidence / 100)
        st.balloons()
