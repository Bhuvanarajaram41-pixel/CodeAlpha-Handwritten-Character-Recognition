# Handwritten Character Recognition using Deep Learning

A deep learning application that recognizes handwritten digits and letters using a Convolutional Neural Network (CNN). Built with Python, TensorFlow/Keras, and Streamlit, the application allows users to upload handwritten character images and receive model predictions through an interactive web interface.

## Project Overview

Handwritten character recognition is a computer vision task that converts handwritten symbols into machine-readable text. This project extends recognition beyond digits to include uppercase and lowercase English letters.

The trained model supports **62 alphanumeric classes**:

- Digits: `0–9`
- Uppercase letters: `A–Z`
- Lowercase letters: `a–z`

## Key Features

- CNN-based handwritten character classification
- Support for 62 alphanumeric classes
- Image preprocessing before prediction
- Predicted character and confidence score
- Top-5 prediction results
- Interactive Streamlit interface
- Trained model integration using TensorFlow/Keras

## Technology Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Computer Vision:** Image preprocessing
- **Web Interface:** Streamlit
- **Data Handling:** NumPy
- **Model Format:** `.keras`

## Model Performance

The current model achieved the following results on the test dataset:

| Metric | Result |
|---|---:|
| Test accuracy | 79.47% |
| Macro F1-score | 0.7875 |
| Number of classes | 62 |

*Performance values are based on the project's reported evaluation results.*

## Repository Structure

```text
CodeAlpha-Handwritten-Character-Recognition/
├── app.py
├── character_classes.json
├── handwritten_character_cnn.keras
├── handwritten_digit_cnn.keras
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Bhuvanarajaram41-pixel/CodeAlpha-Handwritten-Character-Recognition.git
cd CodeAlpha-Handwritten-Character-Recognition
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

## How It Works

1. The user uploads a handwritten character image.
2. The application preprocesses the image into the format expected by the trained model.
3. The CNN evaluates the image and generates class predictions.
4. The application displays the predicted character, confidence score, and top-5 predictions.

## Live Demo

[Try the deployed application](https://codealpha-handwritten-character-recognition-ewprevgcznwa7jgxni.streamlit.app/)

## Future Improvements

- Improve performance on visually similar characters.
- Expand testing across different handwriting styles.
- Add support for recognizing complete words and lines of text.
- Explore model optimization for faster inference.

## Disclaimer

This project is developed for educational and learning purposes. Predictions may be incorrect, particularly for unclear handwriting or visually similar characters.

## Author

**Bhuvaneshwari R**

- [GitHub](https://github.com/Bhuvanarajaram41-pixel)
- [Project Repository](https://github.com/Bhuvanarajaram41-pixel/CodeAlpha-Handwritten-Character-Recognition)

---

Developed as part of machine learning project work.
