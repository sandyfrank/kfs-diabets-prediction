import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Charger le modèle
model = load_model('segmentation_model.h5')

# Prétraitement de l'image
def preprocess_image(image, img_size=(256, 256)):
    image = cv2.resize(image, img_size)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Fonction pour superposer le masque sur l'image
def overlay_mask_on_image(original_image, mask, color=(0, 255, 0), alpha=0.5):
    # Redimensionner le masque à la taille de l'image originale
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))

    # Appliquer une couleur au masque (ici, vert)
    mask_colored = np.zeros_like(original_image)
    mask_colored[:, :] = color

    # Masque binaire
    mask_colored = cv2.bitwise_and(mask_colored, mask_colored, mask=mask_resized)

    # Superposer le masque sur l'image originale avec un certain alpha pour la transparence
    image_with_mask = cv2.addWeighted(original_image, 1 - alpha, mask_colored, alpha, 0)
    return image_with_mask

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

    # Superposer le masque sur l'image originale
    image_with_mask = overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.5)  # Green color with transparency

    # Affichage des résultats
    st.image(image_with_mask, caption="Image avec Masque Segmenté", use_column_width=True)
