import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image

# Load the trained model
model = keras.models.load_model("./drowsiness_model.h5")

# Define class labels
class_labels = ["Drowsy", "Non-Drowsy"]

# Function to preprocess image
def preprocess_image(image):
    IMG_SIZE = 128  # Same size as the model's input
    
    # Convert RGBA (4 channels) to RGB (3 channels)
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # Resize and normalize the image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    return image

# Streamlit UI
st.title("🚗 Driver Drowsiness Detection")
st.write("Upload an image to check if the driver is **Drowsy** or **Non-Drowsy**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)[0][0]
    predicted_class = class_labels[int(prediction > 0.5)]  # Convert probability to class

    # Display result
    st.subheader(f"Prediction: {predicted_class}")

    # Show confidence score
    confidence = prediction if prediction > 0.5 else 1 - prediction
    st.write(f"Confidence: {confidence:.2%}")
