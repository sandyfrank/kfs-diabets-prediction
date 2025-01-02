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
#############################
# Define the pages of the app
def home_page():
    
    st.title("Welcome to Diabetes Prediction")
    st.write("This app helps predict the likelihood of diabetes based on various inputs.")
    st.write("Use the sidebar to navigate between the home page and educational resources.")

def education_page():
    st.title("Education about Diabetes")
    
    # Section 1: General Information
    st.header("1. Généralités")
    st.write("""
    Le diabète se caractérise par une hyperglycémie chronique due à un défaut de sécrétion ou d’assimilation de l’insuline. L’insuline, seule hormone hypoglycémiante, est produite par les cellules béta du pancréas.

    La norme de la glycémie est comprise entre 0,8 et 1,10 g/L. On parle de diabète lorsque la glycémie à jeun est supérieure à 1,26 g/L à 2 reprises, ou par une glycémie supérieure à 2g/L à n’importe quel moment de la journée.

    Il existe trois types de diabète :
    - Le diabète de type 1
    - Le diabète de type 2
    - Le diabète gestationnel
    """)

    # Section 2: Type 1 Diabetes
    st.header("2. Le diabète de type 1")
    st.write("""
    Maladie auto-immune caractérisée par une hyperglycémie chronique due à une destruction progressive des cellules béta du pancréas. Il concerne environ 10 % des patients (environ 150 000 personnes), plutôt les enfants et adolescents ou les adultes de moins de 40 ans.
    
    ### 2.1- Signes cliniques
    Signes d’apparition brutale, qui débutent lorsque que 80% des cellules sont détruites:
    - Polyurie
    - Polydipsie
    - Perte de poids
    - Sensation de faim fréquente = polyphagie
    - Asthénie
    - Troubles visuels
    - Haleine cétonique
    - Acidose retrouvée dans les gaz du sang (pH < 7.3 en artériel, < 7,25 en veineux et bicarbonates < 22 mmol/L)
    - Glycosurie si la glycémie > 1,8 g/L

    ### 2.2- Traitement
    - Insulinothérapie à vie
    - Surveillance glycémique plusieurs fois par jour
    - Surveillance de l’HbA1c (hémoglobine glyquée : norme = < 7%)
    - Régime alimentaire adapté
    - Activité sportive
    """)

    # Section 3: Type 2 Diabetes
    st.header("3. Le diabète de type 2")
    st.write("""
    Maladie d’évolution lente, caractérisée par une insulinorésistance des cellules entraînant une hyperglycémie chronique. Les facteurs favorisants sont : une prédisposition génétique, une mauvaise alimentation, un surpoids, un manque d’activité physique, une mauvaise hygiène de vie, antécédents de diabète gestationnel. Il concerne 90% des patients (environ 3 millions de personnes), d’âge plutôt avancé (entre 40 et 50 ans).
    
    ### 3.1- Signes Cliniques
    Découverte le plus souvent fortuite, ou à l’occasion d’une complication (IDM, AVC…)
    - Hyperglycémie
    - Apparition de complications diverses

    ### 3.2- Traitement
    - Régime alimentaire adapté
    - Activité sportive si possible
    - Antidiabétiques oraux
    - Insulinothérapie
    - Surveillance glycémique plusieurs fois par jour
    - Surveillance de l’HbA1c (hémoglobine glyquée)
    """)

    # Section 4: Diabetes Complications
    st.header("4. Les complications du diabète")
    st.write("""
    - Hypoglycémie (signes : sueurs, pâleur, fringale, tremblements, sensation de malaise, troubles de la vision et de la concentration, somnolence, coma)
    - Hyperglycémie (signes : asthénie, bouche sèche, polyurie, soif intense)
    - La rétinopathie diabétique (cause de cécité)
    - La néphropathie diabétique (entraînant une insuffisance rénale terminale)
    - Infections urinaires plus fréquentes
    - Neuropathies périphérique et végétative diabétique (atteinte des nerfs)
    - Macro angiopathies (coronaropathie, atteinte carotidienne, artériopathie des membres inférieurs,…)
    - Pied diabétique (mal perforant plantaire)
    - Coma diabétique
    """)

    # Section 5: Hygienic-Dietetic Rules
    st.header("5. Règles hygiéno-diététiques")
    st.write("""
    - Alimentation équilibrée où chaque groupe d’aliments doit être représenté
    - Pratique sportive régulière
    - Arrêt du tabac 
    - Eviter la prise de poids
    - Surveillance de toute plaie, même minime
    """)

    # Section 6: Role of Nurses (IDE)
    st.header("6. Rôle IDE")
    st.write("""
    - Réalisation des glycémies capillaires
    - Préparation et injection d’insulines
    - Adaptation des doses d’insulines
    - Education des patients : réalisation des glycémies capillaires, injections d’insuline, adaptation des doses, signes d’hypo et d’hyperglycémie, régime alimentaire, expliquer l’importance de noter les glycémies et doses d’insuline réalisées dans un carnet d’autosurveillance.
    - Réalisation de bandelettes urinaires pour contrôle de la cétonurie
    - Surveillance de l’apparition des complications
    """)

    # Section 7: Gestational Diabetes
    st.header("7. Diabète gestationnel")
    st.write("""
    Selon la définition de l’OMS, le diabète gestationnel est un trouble de la tolérance glucidique conduisant à une hyperglycémie de sévérité variable, débutant ou diagnostiqué pour la première fois pendant la grossesse. Les signes cliniques sont les mêmes que les autres types de diabète. Dans 90% des cas le diabète gestationnel disparaît quelques semaines après l’accouchement.

    ### 7.1- Facteurs de risque
    - Grossesse tardive
    - Obésité ou surpoids de la mère
    - Antécédents de diabète gestationnel
    - Antécédents familiaux de diabète de type 2
    - Antécédents de macrosomie fœtale

    ### 7.2- Complications pour l’enfant
    - Macrosomie (poids de naissance > à 4kg)
    - Accouchement difficile
    - Détresse respiratoire
    - Hypoglycémie néonatale
    - Risque accrue de diabète de type 2 à l’âge adulte 

    ### 7.3- Complications pour la mère
    - Fausses couches
    - Accouchement par césarienne
    - Risque accrue de prééclampsie (prise de poids, œdème, HTA)
    - Risque de développer un diabète de type 2 après la grossesse
    - Risque d’accouchement prématuré

    ### 7.4- Dépistage
    Consiste en la réalisation du test HGPO (Hyperglycémie Provoquée par voie Orale) à 75 g de glucose.

    ### 7.5- Prévention
    - Alimentation équilibrée dès le début de la grossesse
    - Activité physique régulière en l’absence de contre indication
    - Limiter les apports glycémiques trop importants

    ### 7.6- Traitement
    - Régime diététique hypocalorique et limitant l’apport glycémique
    - Insulinothérapie si régime inefficace
    """)


def prediction_page():
    # Input form with interactive widgets
    st.markdown("<div class='form-container'>", unsafe_allow_html=True)
    
    gender = st.radio("Gender (0 for Female, 1 for Male)" if language == "English" else "Genre (0 pour Femme, 1 pour Homme)", [0, 1])
    age = st.slider("Age" if language == "English" else "Âge", 0, 120, 30)
    hypertension = st.radio("Hypertension (0 for No, 1 for Yes)" if language == "English" else "Hypertension (0 pour Non, 1 pour Oui)", [0, 1])
    heart_disease = st.radio("Heart Disease (0 for No, 1 for Yes)" if language == "English" else "Maladie Cardiaque (0 pour Non, 1 pour Oui)", [0, 1])
    smoking_history = st.selectbox("Smoking History (0: Never, 1: No-info, 2: Former, 3: Current, 4: Not-Current)" if language == "English" else "Historique de Tabagisme (0 : Jamais, 1 : Sans Info, 2 : Ancien, 3 : Actuel, 4 : Non Actuel)", [0, 1, 2, 3, 4])
    bmi = st.number_input("BMI (real number between 0 and 100)" if language == "English" else "IMC (nombre réel entre 0 et 100)", 0.0, 100.0, 25.0)
    HbA1c_level = st.number_input("HbA1c Level (real number between 0 and 100)" if language == "English" else "Niveau d'HbA1c (nombre réel entre 0 et 100)", 0.0, 100.0, 5.7)
    blood_glucose_level = st.number_input("Blood Glucose Level in mg/dL (real number between 0 and 1000)" if language == "English" else "Niveau de Glucose Sanguin en mg/dL (nombre réel entre 0 et 1000)", 0.0, 1000.0, 100.0)
    
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


# Language selection
language = st.sidebar.selectbox("Select Language / Sélectionnez la langue", ["English", "Français"])

# Title and subtitle
if language == "English":
    st.markdown("<div class='main-title'>KFS AI4Health Diabetes Prediction using AI App</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Enter the following details to predict diabetes</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='main-title'>Application de KFS AI4Health utilisant l'IA pour la Prédiction du Diabète</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Entrez les informations suivantes pour prédire le diabète</div>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Home", "Education about Diabetes", "Diabetes Prediction"])

# Call the appropriate function based on page selection
if page == "Home":
    home_page()
elif page == "Education about Diabetes":
    education_page()
elif page == "Diabetes Prediction":
    prediction_page()

