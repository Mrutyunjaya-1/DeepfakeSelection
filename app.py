import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# MUST be the first Streamlit command
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🛡️",
    layout="centered"
)

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5", compile=False)

model = load_model()

IMG_SIZE = 224

st.title("🛡️ Deepfake Detection System")
st.write("Upload an image to check whether it is REAL or FAKE.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    image = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(
        img,
        verbose=0
    )[0][0]

    st.subheader("Prediction Result")

    if prediction > 0.65:
        st.error(
            f"🚨 FAKE (Confidence: {prediction:.2f})"
        )
    else:
        st.success(
            f"✅ REAL (Confidence: {1-prediction:.2f})"
        )

    st.write(
        "Raw Confidence Score:",
        round(float(prediction), 4)
    )
