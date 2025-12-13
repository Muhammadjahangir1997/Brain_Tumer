import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os
import tensorflow as tf # TensorFlow import karna zaroori hai

# --- Model Variables ---
MODEL_PATH = "brain_tumor_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1B9vbxl1vg0bFhVA_5SReq9J_aocJ46BA"

# Caching the model loading process
# st.cache_resource is best for ML models
@st.cache_resource
def load_and_cache_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model, please wait...")
        # 1. Download Attempt
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Download Error: Could not retrieve file from Google Drive. {e}")
            st.stop() # Agar download fail ho toh ruk jao

    # 2. Loading Attempt with Error Handling
    try:
        # Load the model
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        # Agar loading fail ho toh file ko delete kar do aur user ko batao
        st.error(f"Model Load Error: The downloaded model file seems corrupt. Error: {e}")
        st.warning(f"Deleting the corrupt file '{MODEL_PATH}'. Please refresh the app to re-download.")
        
        # Corrupt file ko delete kar dein taki Streamlit next time naya download kare
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        
        st.stop() # App ko rok do

# Load trained model
model = load_and_cache_model()

# ... rest of your Streamlit UI code
