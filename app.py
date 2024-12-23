import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load('best_xgb_model.joblib')
scalar = joblib.load('scaler.joblib')

# Streamlit HTML customization
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
            font-family: 'Arial', sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 3em;
            color: #4CAF50;
            font-weight: bold;
            margin-top: 20px;
        }
        .sub-title {
            text-align: center;
            font-size: 1.5em;
            color: #666;
            margin-bottom: 30px;
        }
        .form-container {
            background: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        .button-style {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .button-style:hover {
            background-color: #45a049;
        }
        .result {
            font-size: 1.2em;
            color: #333;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Language selection
language = st.sidebar.selectbox("Select Language / Sélectionnez la langue", ["English", "Français"])

# Title and subtitle
if language == "English":
    st.markdown("<div class='main-title'>KFS AI4Health Diabetes Prediction using AI App</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Enter the following details to predict diabetes</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='main-title'>Application de KFS AI4Health utilisant l'IA pour la Prédiction du Diabète</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Entrez les informations suivantes pour prédire le diabète</div>", unsafe_allow_html=True)

# Input form
st.markdown("<div class='form-container'>", unsafe_allow_html=True)

gender_label = "Gender (0 for Female, 1 for Male)" if language == "English" else "Genre (0 pour Femme, 1 pour Homme)"
age_label = "Age" if language == "English" else "Âge"
hypertension_label = "Hypertension (0 for No, 1 for Yes)" if language == "English" else "Hypertension (0 pour Non, 1 pour Oui)"
heart_disease_label = "Heart Disease (0 for No, 1 for Yes)" if language == "English" else "Maladie Cardiaque (0 pour Non, 1 pour Oui)"
smoking_history_label = "Smoking History (0: Never, 1: No-info, 2: Former, 3: Current, 4: Not-Current)" if language == "English" else "Historique de Tabagisme (0 : Jamais, 1 : Sans Info, 2 : Ancien, 3 : Actuel, 4 : Non Actuel)"
bmi_label = "BMI (real number between 0 and 100)" if language == "English" else "IMC (nombre réel entre 0 et 100)"
HbA1c_level_label = "HbA1c Level (real number between 0 and 100)" if language == "English" else "Niveau d'HbA1c (nombre réel entre 0 et 100)"
blood_glucose_label = "Blood Glucose Level in mg/dL (real number between 0 and 1000)" if language == "English" else "Niveau de Glucose Sanguin en mg/dL (nombre réel entre 0 et 1000)"

gender = st.number_input(gender_label, 0, 1)
age = st.number_input(age_label, 0, 200)
hypertension = st.number_input(hypertension_label, 0, 1)
heart_disease = st.number_input(heart_disease_label, 0, 1)
smoking_history = st.number_input(smoking_history_label, 0, 4)
bmi = st.number_input(bmi_label, 0.0, 100.0)
HbA1c_level = st.number_input(HbA1c_level_label, 0.0, 100.0)
blood_glucose_level = st.number_input(blood_glucose_label, 0.0, 1000.0)

st.markdown("</div>", unsafe_allow_html=True)

# Prepare input data for prediction
input_data = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])
input_data = scalar.transform(input_data)

# Button and results
if st.button("Diabetes Diagnostic Result" if language == "English" else "Résultat du Diagnostic du Diabète", key="predict"):
    prediction = model.predict(input_data)

    st.markdown("<div class='form-container'>", unsafe_allow_html=True)

    if prediction[0] == 1:
        if language == "English":
            st.markdown("<p class='result'>According to the AI, we are really sorry to say but it seems like you are Diabetic, so it's urgent to consult a doctor.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='result'>D'après l'IA, Nous sommes vraiment désolés, mais il semble que vous soyez diabétique., il est donc urgent de consulter un médecin.</p>", unsafe_allow_html=True)
    else:
        if language == "English":
            st.markdown("<p class='result'>According to the AI, this person is not diabetic.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='result'>D'après l'IA, cette personne n'est pas diabétique.</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
