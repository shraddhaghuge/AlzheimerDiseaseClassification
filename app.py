import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("alzheimer_resnet50_model.h5")

# Class labels
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Preprocessing function
def preprocess_image(img):
    img = img.convert('RGB')  # Ensure 3 channels
    img = img.resize((176, 176))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit UI
st.title("🧠 Alzheimer's MRI Classifier")
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded MRI Image', use_column_width=True)

    with st.spinner('Predicting...'):
        processed_img = preprocess_image(img)
        predictions = model.predict(processed_img)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    # Show full confidence scores
    st.subheader("Confidence for All Classes")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {prob * 100:.2f}%")
