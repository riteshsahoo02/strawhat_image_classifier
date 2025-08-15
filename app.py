import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("strawhat_cnn.h5")
    return model

# Load class names
def load_class_names():
    with open("class_names.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

model = load_model()
class_names = load_class_names()

st.title("Strawhat Crew Classifier 🏴‍☠️")
st.write("Upload an image of any Strawhat crew member, and I’ll tell you who it is!")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for model
    img_size = (128, 128)  # Change if your model uses different input size
    img_array = np.array(image.resize(img_size)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    st.success(f"**Prediction:** {predicted_class} ({confidence:.2f}% confidence)")
