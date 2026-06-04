import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5", compile=False)

model = load_model()

IMG_SIZE = 224

st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Deepfake Detection System")
st.write("Upload an image to check whether it is REAL or FAKE.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image and convert to RGB
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(image, caption="Uploaded Image", width=300)

    # Resize image
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to numpy array
    img = np.array(image)

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)[0][0]

    st.subheader("Prediction Result")

    if prediction > 0.65:
        st.error(f"🚨 FAKE (Confidence: {prediction:.2f})")
    else:
        st.success(f"✅ REAL (Confidence: {1 - prediction:.2f})")

    st.write("Raw Confidence Score:", round(float(prediction), 4))
