import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "handwritten_character_cnn.keras"
    )


# ---------------------------------------------------------
# LOAD CLASS NAMES
# ---------------------------------------------------------

@st.cache_data
def load_classes():
    with open(
        "character_classes.json",
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


model = load_model()
class_names = load_classes()


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------

def preprocess_image(image):
    img = np.array(image)

    # Convert RGB/RGBA image to grayscale
    if len(img.shape) == 3:

        if img.shape[2] == 4:
            img = cv2.cvtColor(
                img,
                cv2.COLOR_RGBA2GRAY
            )

        else:
            img = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2GRAY
            )

    # Resize to model input size
    img = cv2.resize(
        img,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    # Normalize pixels
    img = img.astype("float32") / 255.0

    # Add batch and channel dimensions
    img = img.reshape(
        1,
        28,
        28,
        1
    )

    return img


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("✍️ Handwritten Character Recognition")

st.markdown(
    """
    ### Recognize handwritten digits and English characters using CNN

    Upload an image containing a handwritten **digit or English
    alphabet character**, and the trained deep learning model
    will predict the character.
    """
)

st.divider()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("📊 Model Information")

    st.write("**Model:** Convolutional Neural Network")

    st.write("**Classes:** 62")

    st.write(
        "**Classes include:** "
        "0–9, A–Z, a–z"
    )

    st.write(
        "**Test Accuracy:** 79.47%"
    )

    st.write(
        "**Macro F1 Score:** 0.7875"
    )

    st.divider()

    st.info(
        "For best results, upload a clear image "
        "containing one handwritten character."
    )


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload a handwritten character image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📷 Uploaded Image")

        st.image(
            image,
            caption="Uploaded handwritten character",
            width=300
        )

    # Preprocess
    processed_image = preprocess_image(image)

    with col2:

        st.subheader("🔲 Processed Image")

        processed_display = (
            processed_image[0, :, :, 0]
        )

        st.image(
            processed_display,
            caption="28 × 28 model input",
            width=200,
            clamp=True
        )


    # Make prediction
    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]


    # Get top prediction
    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_character = class_names[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    ) * 100


    st.divider()


    # -----------------------------------------------------
    # MAIN RESULT
    # -----------------------------------------------------

    st.subheader("🎯 Prediction")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Predicted Character",
            predicted_character
        )

    with result_col2:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


    # -----------------------------------------------------
    # TOP 5 PREDICTIONS
    # -----------------------------------------------------

    st.subheader("🔝 Top 5 Predictions")

    top_indices = np.argsort(
        predictions
    )[-5:][::-1]

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        character = class_names[index]

        probability = (
            float(predictions[index]) * 100
        )

        st.write(
            f"**{rank}. {character}** — "
            f"{probability:.2f}%"
        )

        st.progress(
            min(probability / 100, 1.0)
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "CodeAlpha Machine Learning Internship • "
    "Handwritten Character Recognition"
)