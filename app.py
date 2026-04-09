import streamlit as st
import joblib
import numpy as np

st.title("Retail Demand Forecasting")

model = joblib.load("model/demand_model.pkl")
scaler = joblib.load("model/scaler.pkl")

lag_1 = st.number_input("Demand Yesterday (Lag_1)", min_value=0.0, value=1000.0)
lag_7 = st.number_input("Demand 7 Days Ago (Lag_7)", min_value=0.0, value=950.0)
rolling_7 = st.number_input("7-Day Average Demand (Rolling_7)", min_value=0.0, value=980.0)

input_data = np.array([[lag_1, lag_7, rolling_7]])
input_scaled = scaler.transform(input_data)

if st.button("Predict Demand"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Demand: {prediction:.2f}")