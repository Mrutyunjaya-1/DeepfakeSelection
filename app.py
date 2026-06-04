import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model once (faster)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

IMG_SIZE = 224

st.title("🛡️ Deepfake Detection System")
st.write("Upload an image to check if it is REAL or FAKE")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # ensures 3 channels

    # Display image
    st.image(image, caption="Uploaded Image", width=300)

    # Convert to numpy
    img = np.array(image)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)[0][0]

    # Result
    if prediction > 0.65:
        st.error(f"🚨 FAKE (Confidence: {prediction:.2f})")
    else:
        st.success(f"✅ REAL (Confidence: {1 - prediction:.2f})")

    st.write("Confidence Score:", float(prediction))