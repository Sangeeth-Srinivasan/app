import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.image import img_to_array

# Constants
IMG_SIZE = (100, 100)
CLASS_NAMES = ['banana_pack', 'banana_unpack', 'capsicum_pack', 'capsicum_unpack',
               'carrot_pack', 'carrot_unpack', 'cucumber_pack', 'cucumber_unpack',
               'grapes_pack', 'grapes_unpack', 'kiwi_pack', 'kiwi_unpack',
               'lemon_pack', 'lemon_unpack', 'onion_pack', 'onion_unpack']

# **Cached Model Loaders**
@st.cache_resource
def load_ml_models():
    return {
        "KNN": joblib.load("decision_tree.pkl"),
        "Naive Bayes": joblib.load("naive_bayes.pkl"),
        "Random Forest": joblib.load("logistic_regression.pkl"),
    }

@st.cache_resource
def load_dl_models():
    return {
        "Custom CNN": tf.keras.models.load_model("cnn_model.h5"),
        "MobileNetV2": tf.keras.models.load_model("mobilenetv2.h5"),
        "VGG16": tf.keras.models.load_model("vgg16.h5"),
    }

# Load models once
ml_models = load_ml_models()
dl_models = load_dl_models()

# **Preprocessing Functions**
def preprocess_image(image):
    """Preprocesses the image for Deep Learning models."""
    image.seek(0)  # Reset file pointer
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Error: Unable to read the image. Please upload a valid file.")

    img = cv2.resize(img, IMG_SIZE)
    img = img_to_array(img) / 255.0  # Normalize
    return img

def preprocess_flattened(image):
    """Preprocesses the image for Machine Learning models."""
    img = preprocess_image(image)
    return img.flatten().reshape(1, -1)  # Flatten for ML models

# **Streamlit UI**
st.title("Fruit and Veg Package Classification Prediction 🍌🥒🍇🍋")
st.write("Upload an image of a fruit, and the trained models will predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    try:
        # Preprocess Image
        img_array = preprocess_image(uploaded_file)
        img_expanded = np.expand_dims(img_array, axis=0)  # Batch dimension
        img_flattened = preprocess_flattened(uploaded_file)

        # **ML Model Predictions**
        st.subheader("Machine Learning Model Predictions")
        for model_name, model in ml_models.items():
            pred = model.predict(img_flattened)[0]  # Extract first prediction
            predicted_class = CLASS_NAMES[int(pred)]  # Ensure correct indexing
            st.write(f"**{model_name}:** {predicted_class}")

        # **Deep Learning Model Predictions**
        st.subheader("Deep Learning Model Predictions")
        for model_name, model in dl_models.items():
            pred = model.predict(img_expanded)
            predicted_class = CLASS_NAMES[np.argmax(pred)]  # Argmax is correct for DL
            st.write(f"**{model_name}:** {predicted_class}")

        st.success("Prediction Completed!")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
