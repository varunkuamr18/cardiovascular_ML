import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# --- 1. SETTINGS & ASSET LOADING ---
st.set_page_config(page_title="CardioCheck AI", layout="wide")

# We use st.cache_resource to load the model only once
@st.cache_resource
def load_model_assets():
    try:
        # Try loading the files from the current directory
        m = joblib.load('cardio_model.pkl')
        s = joblib.load('scaler.pkl')
        return m, s
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None

# Initialize variables
model, scaler = load_model_assets()

# --- 2. UI ---
st.title("🏥 Cardiovascular Disease Prediction Portal")

if model is None or scaler is None:
    st.warning("⚠️ Waiting for model files... please ensure 'cardio_model.pkl' and 'scaler.pkl' are uploaded to GitHub.")
    st.stop() # Stops the script here so line 47 is never reached if scaler is missing

# --- 3. INPUT FORM ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (Years)", 1, 120, 50)
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x==1 else "Male")
        height = st.number_input("Height (cm)", 50, 250, 165)
        weight = st.number_input("Weight (kg)", 20.0, 300.0, 70.0)
    with col2:
        ap_hi = st.number_input("Systolic BP", 50, 250, 120)
        ap_lo = st.number_input("Diastolic BP", 30, 150, 80)
        chol = st.selectbox("Cholesterol", [1, 2, 3])
        gluc = st.selectbox("Glucose", [1, 2, 3])
    
    smoke = st.toggle("Smoker")
    alco = st.toggle("Alcohol")
    active = st.toggle("Active", value=True)
    
    submit = st.form_submit_button("Generate Report")

# --- 4. PREDICTION ---
if submit:
    # Use 365.25 for better accuracy
    input_data = np.array([[age*365.25, gender, height, weight, ap_hi, ap_lo, chol, gluc, int(smoke), int(alco), int(active)]])
    
    # This line won't crash now because we used st.stop() above if scaler is None
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    prob = model.predict_proba(scaled_data)[0][1]

    st.divider()
    if prediction[0] == 1:
        st.error(f"### High Risk: {prob:.1%}")
    else:
        st.success(f"### Low Risk: {prob:.1%}")
