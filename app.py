import streamlit as st
import pandas as pd
import joblib
import numpy as np

# This version is "Path-Safe" for GitHub
st.set_page_config(page_title="Cardio Predictor")

@st.cache_resource
def load_model():
    # Loading locally from the GitHub root folder
    model = joblib.load('cardio_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
except Exception as e:
    st.error("Model files not found. Ensure 'cardio_model.pkl' and 'scaler.pkl' are uploaded to GitHub.")
    st.stop()

st.title("Cardiovascular Disease Prediction")

# Example simple input for testing
age = st.number_input("Age (Years)", 18, 100, 50)
if st.button("Run Test Prediction"):
    # This matches the 11 features expected by your model
    test_data = np.array([[age*365, 1, 165, 70, 120, 80, 1, 1, 0, 0, 1]])
    scaled_data = scaler.transform(test_data)
    result = model.predict(scaled_data)
    st.write(f"Prediction: {'Cardiovascular Disease Detected' if result[0]==1 else 'No Disease Detected'}")
