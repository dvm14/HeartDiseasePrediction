import streamlit as st
import requests

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient information below and click **Predict** to see the model's output.")

API_URL = "https://heart-api-41967139984.us-central1.run.app/predict"


# --- Input fields ---
Age = st.number_input("Age", min_value=1, max_value=120, value=52)
Sex = st.selectbox("Sex", ["M", "F"])
ChestPainType = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=125)
Cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=212)
FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
RestingECG = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
MaxHR = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=168)
ExerciseAngina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
Oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict ❤️"):
    data = {
        "Age": Age,
        "Sex": Sex,
        "ChestPainType": ChestPainType,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "RestingECG": RestingECG,
        "MaxHR": MaxHR,
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": Oldpeak,
        "ST_Slope": ST_Slope
    }

    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            result = response.json()["prediction"]
            if result == 1:
                st.error("The model predicts Heart Disease.")
            else:
                st.success("The model predicts No Heart Disease.")
        else:
            st.warning(f"Unexpected response ({response.status_code}): {response.text}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
