import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd






# Load my model 

model = joblib.load('best_xgb_model.joblib')
scalar = joblib.load('scaler.joblib')


# Streamlit

st.title('KFS AI4Health Diabetes Prediction App')

st.write('Enter the following details to predict diabetes')

# Features

gender = st.number_input("Gender, 0 for female and 1 for male ",0,1)

age = st.number_input("Age", 0, 200)

hypertension = st.number_input("hypertension, 0 for NO and 1 for Yes ", 0, 1)

heart_disease = st.number_input("heart_disease, 0 for NO and 1 for Yes", 0, 1)

smoking_history = st.number_input("smoking_history, 0 for never, 1 for No info, 2 for former, 3 for current and 4 for not current ", 0, 4)

bmi = st.number_input("bmi, real number between 0 and 20 ", 0, 20)

HbA1c_level = st.number_input("HbA1c_level,  real number between 0 and 20", 0, 20)

blood_glucose_level = st.number_input("blood_glucose_level, real number between 0 and 300 ", 0, 300)



#prepare input for prediction

input_data = np.array([[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]])

input_data = scalar.transform(input_data)


if st.button('Diabete diagnostic result'):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.write("EN: According to the AI, this person has a very high probability of being diabetic, so it's urgent to consult a doctor.")
        st.write("FR: D'après l'IA, cette personne a une très forte probabilité d'être diabétique, il est donc urgent de consulter un médecin.")
    else:
        st.write("EN: According to the AI, this person is not diabetic")
        st.write("FR: D'après l'IA, cette personne n'est pas diabétique")