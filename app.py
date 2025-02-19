import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load Model
model = joblib.load("best_random_forest_model.pkl")

st.title("Heart Failure Prediction Input Form")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=60)
anaemia = st.selectbox("Anaemia", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", min_value=0, value=200)
diabetes = st.selectbox("Diabetes", [0, 1])
ejection_fraction = st.number_input("Ejection Fraction", min_value=0, max_value=100, value=50)
high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1])
platelets = st.number_input("Platelets", min_value=0.0, value=250000.0, format="%.2f")
serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, value=1.0, format="%.2f")
serum_sodium = st.number_input("Serum Sodium", min_value=100, max_value=200, value=140)
sex = st.selectbox("Sex (Male=1, Female=0)", [0, 1])
smoking = st.selectbox("Smoking", [0, 1])
time = st.number_input("Time (Follow-up period)", min_value=0, value=100)

# Preprocessing
def preprocess_input(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure,
                    platelets, serum_creatinine, serum_sodium, sex, smoking, time):
    input_data = pd.DataFrame({
        'age': [age],
        'anaemia': [anaemia],
        'creatinine_phosphokinase': [creatinine_phosphokinase],
        'diabetes': [diabetes],
        'ejection_fraction': [ejection_fraction],
        'high_blood_pressure': [high_blood_pressure],
        'platelets': [platelets],
        'serum_creatinine': [serum_creatinine],
        'serum_sodium': [serum_sodium],
        'sex': [sex],
        'smoking': [smoking],
        'time': [time]
    })
    
    # Handle outliers using IQR
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    input_data = remove_outliers(input_data, 'creatinine_phosphokinase')
    input_data = remove_outliers(input_data, 'serum_creatinine')
    input_data = remove_outliers(input_data, 'platelets')
    
    return input_data

# Prediction
if st.button("Predict Heart Failure Risk"):
    processed_data = preprocess_input(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, 
                                      high_blood_pressure, platelets, serum_creatinine, serum_sodium, 
                                      sex, smoking, time)
    prediction = model.predict(processed_data)
    result = "Gagal Jantung" if prediction[0] == 1 else "Tidak Gagal Jantung"
    st.write(result)
