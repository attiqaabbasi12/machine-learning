import streamlit as st
import pandas as pd
import joblib

model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('model_columns.pkl')

st.title("Heart Disease Prediction")
st.markdown("Provide the following details:") 

age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", options=["ATA", "NAP", "ASY", "TA"])
cp_mapping = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
cp = cp_mapping[cp]
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.selectbox("Resting ECG", options=["Normal", "ST", "LVH"])
restecg_mapping = {"Normal": 0, "ST": 1, "LVH": 2}
restecg = restecg_mapping[restecg]
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=["Upsloping", "Flat", "Downsloping"])
slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope = slope_mapping[slope]

if st.button("Predict"):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope'
    ])
    input_data = input_data.reindex(columns=columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.error("The model predicts that you may have heart disease. Please consult a healthcare professional.")
    else:
        st.success("The model predicts that you are unlikely to have heart disease.")
        

