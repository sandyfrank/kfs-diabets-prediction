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

# Title
st.markdown("<div class='main-title'>KFS AI4Health Diabetes Prediction App</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Enter the following details to predict diabetes</div>", unsafe_allow_html=True)

# Input form
st.markdown("<div class='form-container'>", unsafe_allow_html=True)

gender = st.number_input("Gender (0 for Female, 1 for Male)", 0, 1)
age = st.number_input("Age", 0, 200)
hypertension = st.number_input("Hypertension (0 for No, 1 for Yes)", 0, 1)
heart_disease = st.number_input("Heart Disease (0 for No, 1 for Yes)", 0, 1)
smoking_history = st.number_input("Smoking History (0: Never, 1: No-info, 2: Former, 3: Current, 4: Not-Current)", 0, 4)
bmi = st.number_input("BMI (real number between 0 and 20)", 0.0, 20.0)
HbA1c_level = st.number_input("HbA1c Level (real number between 0 and 100)", 0.0, 100.0)
blood_glucose_level = st.number_input("Blood Glucose Level in mg/dL (real number between 0 and 300)", 0.0, 300.0)

st.markdown("</div>", unsafe_allow_html=True)

# Prepare input data for prediction
input_data = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])
input_data = scalar.transform(input_data)

if st.button("Diabetes Diagnostic Result", key="predict"):
    prediction = model.predict(input_data)

    st.markdown("<div class='form-container'>", unsafe_allow_html=True)

    if prediction[0] == 1:
        st.markdown("<p class='result'>EN: According to the AI, this person has a very high probability of being diabetic, so it's urgent to consult a doctor.</p>", unsafe_allow_html=True)
        st.markdown("<p class='result'>FR: D'après l'IA, cette personne a une très forte probabilité d'être diabétique, il est donc urgent de consulter un médecin.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='result'>EN: According to the AI, this person is not diabetic.</p>", unsafe_allow_html=True)
        st.markdown("<p class='result'>FR: D'après l'IA, cette personne n'est pas diabétique.</p>", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
