import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load model and classes
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"D:\major project\best_efficientnet_model.h5")
    return model

def get_class_names():
    return ['Healthy', 'Moderate', 'Severe']  


def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)

    # Handle grayscale
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)

    # Convert to Tensor
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # If grayscale (1 channel), convert to RGB
    if img_tensor.shape[-1] == 1:
        img_tensor = tf.image.grayscale_to_rgb(img_tensor)

    # If RGBA (4 channels), discard alpha
    elif img_tensor.shape[-1] == 4:
        img_tensor = img_tensor[..., :3]

    # Preprocess and expand dims for batch
    img_tensor = preprocess_input(img_tensor)
    img_tensor = tf.expand_dims(img_tensor, axis=0)  # Shape: (1, 224, 224, 3)

    return img_tensor


# Streamlit UI
st.title("Knee Osteoarthritis Classification")
st.write("Upload a knee X-ray image to classify its osteoarthritis level.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    class_names = get_class_names()
    
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### 🧠 Prediction: `{predicted_class}`")
    st.markdown(f"Confidence: `{confidence * 100:.2f}%`")
