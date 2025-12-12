import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
model = load_model("brain_tumor_model.h5")

st.title("Brain Tumor Detection System (AI Powered)")
st.write("Upload a Brain MRI image and the model will predict Tumor or No Tumor")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI Image", width=300)

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
