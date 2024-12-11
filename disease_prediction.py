import streamlit as st
from joblib import load
import numpy as np

st.set_page_config(
    page_title="HSV Disease Prediction",  # Title of your web app 
    layout="wide",                   # Layout options: "centered" or "wide"
    initial_sidebar_state="expanded" # Sidebar options: "expanded" or "collapsed"
)


# Sidebar for Model Selection
st.sidebar.title("Choose Disease")
option = st.sidebar.selectbox(
    "Select a prediction model:",
    ("Diabetes Prediction", "Heart Disease Prediction")
)

# Function to load models
@st.cache_resource

def load_model(file_path):
    return load(file_path)

diabetes_model = load_model('diabetes_model.joblib')
heart_model = load_model('heart_model.joblib')

# Display relevant input fields based on the model chosen
if option == "Diabetes Prediction":
    st.title("Diabetes Prediction")
    
    # Input fields for diabetes prediction
    pregnancies = st.number_input("Pregnancies", min_value=0, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, value=80)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)
    insulin = st.number_input("Insulin Level", min_value=0, value=85)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
    age = st.number_input("Age", min_value=0, value=30)

    # Load the diabetes model
    diabetes_model = load_model('diabetes_model.joblib')

    # Predict button
    if st.button("Predict"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        prediction = diabetes_model.predict(input_data)
        st.write("Prediction:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")

elif option == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")
    
    # Input fields for heart disease prediction
    age = st.number_input("Age", min_value=0, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.number_input("Chest Pain Type", min_value=0, max_value=3, value=1)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0, value=120)
    chol = st.number_input("Cholesterol", min_value=0, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    restecg = st.number_input("Resting Electrocardiographic Results", min_value=0, max_value=2, value=1)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, value=150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, value=1.0)
    slope = st.number_input("Slope of the Peak Exercise", min_value=0, max_value=2, value=1)
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
    thal = st.number_input("Thalassemia", min_value=0, max_value=3, value=2)

    # Convert categorical fields
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    # Load the heart disease model
    heart_model = load_model('heart_model.joblib')

    # Predict button
    if st.button("Predict"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = heart_model.predict(input_data)
        st.write("Prediction:", "Heart Disease Present" if prediction[0] == 1 else "No Heart Disease")

