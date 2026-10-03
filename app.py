import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Handwritten Digit AI",
    page_icon="✍️",
    layout="centered"
)


# ============================================================
# LOAD TRAINED CNN MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "handwritten_digit_cnn.keras"
    )


model = load_model()


# ============================================================
# HANDWRITTEN DIGIT PREPROCESSING
# ============================================================

def preprocess_digit(image_gray):

    # Convert PIL image to NumPy array
    img = np.array(image_gray)

    # Reduce noise
    blur = cv2.GaussianBlur(
        img,
        (5, 5),
        0
    )

    # Convert image to binary
    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    img_h, img_w = binary.shape

    valid_contours = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Ignore regions touching image border
        touches_border = (
            x <= 2 or
            y <= 2 or
            x + w >= img_w - 2 or
            y + h >= img_h - 2
        )

        if not touches_border and area > 10:
            valid_contours.append(contour)

    # No digit detected
    if len(valid_contours) == 0:
        return None

    # Select largest valid contour
    digit_contour = max(
        valid_contours,
        key=cv2.contourArea
    )

    # Get bounding box
    x, y, w, h = cv2.boundingRect(
        digit_contour
    )

    # Crop digit
    digit_crop = binary[
        y:y+h,
        x:x+w
    ]

    # Add padding
    padding = 20

    digit_padded = cv2.copyMakeBorder(
        digit_crop,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # Preserve aspect ratio
    h, w = digit_padded.shape

    scale = 20 / max(h, w)

    new_w = max(
        1,
        int(w * scale)
    )

    new_h = max(
        1,
        int(h * scale)
    )

    digit_resized = cv2.resize(
        digit_padded,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Create 28 × 28 canvas
    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    # Center digit
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = digit_resized

    return canvas


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_digit(canvas):

    # Normalize
    digit_input = (
        canvas.astype("float32") / 255.0
    )

    # Add batch and channel dimensions
    digit_input = digit_input.reshape(
        1,
        28,
        28,
        1
    )

    # CNN prediction
    prediction = model.predict(
        digit_input,
        verbose=0
    )[0]

    predicted_digit = int(
        np.argmax(prediction)
    )

    confidence = float(
        np.max(prediction)
    )

    return predicted_digit, confidence, prediction


# ============================================================
# USER INTERFACE
# ============================================================

st.title("✍️ Handwritten Digit AI")

st.write(
    "Upload an image of a handwritten digit "
    "and let the CNN recognize it."
)

st.info(
    "The model was trained on the MNIST handwritten "
    "digit dataset."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(
        uploaded_file
    ).convert("L")

    st.subheader("📷 Uploaded Image")

    st.image(
        image,
        width=250
    )

    # Preprocess
    canvas = preprocess_digit(
        image
    )

    if canvas is None:

        st.error(
            "No clear handwritten digit was detected. "
            "Please upload another image."
        )

    else:

        # Prediction
        predicted_digit, confidence, probabilities = (
            predict_digit(canvas)
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.subheader("🔮 Prediction")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Digit",
                predicted_digit
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        # ----------------------------------------------------
        # PROCESSED IMAGE
        # ----------------------------------------------------

        st.subheader(
            "🧠 Processed Image (28 × 28)"
        )

        st.image(
            canvas,
            width=180
        )

        # ----------------------------------------------------
        # PROBABILITY DISTRIBUTION
        # ----------------------------------------------------

        st.subheader(
            "📊 Prediction Probabilities"
        )

        probability_data = {
            str(i): float(probabilities[i])
            for i in range(10)
        }

        st.bar_chart(
            probability_data
        )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        st.success(
            f"The CNN predicts this handwritten digit "
            f"as **{predicted_digit}** with "
            f"**{confidence * 100:.2f}% confidence**."
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("ℹ️ About the Model")

    st.write(
        """
        **Model:** Convolutional Neural Network (CNN)

        **Dataset:** MNIST

        **Classes:** 10 digits (0–9)

        **Input Size:** 28 × 28 pixels

        **Test Accuracy:** 99.07%
        """
    )

    st.divider()

    st.write(
        "Built as part of the CodeAlpha "
        "Machine Learning Internship."
    )