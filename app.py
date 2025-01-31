import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
MODEL_PATH = "pneumonia_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define function to preprocess image
def preprocess_image(img):
    img = img.convert("RGB")  # Convert grayscale to RGB
    img = img.resize((150, 150))  # Resize image to match model input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Streamlit UI
st.title("Pneumonia Detection using AI 🩺")
st.write("Upload a chest X-ray image to check for Pneumonia.")

# Upload Image
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)[0][0]  # Get model prediction
    
    # Show Result
    st.subheader("Prediction:")
    if prediction > 0.5:
        st.error("🚨 Pneumonia Detected! Consult a doctor immediately.")
    else:
        st.success("✅ No Pneumonia detected. The lungs appear normal.")

# Footer
st.write("Developed with ❤️ using Streamlit & TensorFlow")

