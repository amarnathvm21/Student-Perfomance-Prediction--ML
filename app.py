import streamlit as st
import pandas as pd
import joblib

model = joblib.load("performance_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("features.pkl")

st.title("Student Performance Prediction")

hours_studied = st.number_input("Hours Studied", 0, 15)
previous_scores = st.number_input("Previous Scores", 0, 100)
sleep_hours = st.number_input("Sleep Hours", 0, 15)
extra = st.selectbox("Extracurricular Activities", ["No", "Yes"])
papers = st.number_input("Sample Question Papers Practiced", 0, 20)

input_data = {
    "Hours Studied": hours_studied,
    "Previous Scores": previous_scores,
    "Sleep Hours": sleep_hours,
    "Extracurricular Activities": 1 if extra == "Yes" else 0,
    "Sample Question Papers Practiced": papers
}

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

if st.button("Predict"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    st.success(f"Predicted Performance Index: {prediction[0]:.2f}")
