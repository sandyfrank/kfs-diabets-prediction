
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Charger le modèle
model = load_model('segmentation_model.h5')

# Prétraitement de l'image
def preprocess_image(image, img_size=(256, 256)):
    image = cv2.resize(image, img_size)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Interface Streamlit
st.title("Segmentation des Tumeurs du Sein")
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image Originale", use_column_width=True)
    image = np.array(image)

    # Prétraitement et prédiction
    processed_image = preprocess_image(image)
    mask = model.predict(processed_image)
    mask = (mask[0] > 0.5).astype(np.uint8)

    # Affichage des résultats
    st.image(mask, caption="Masque Segmenté", use_column_width=True)
