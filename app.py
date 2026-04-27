import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and scaler (Ensure these files are in your GitHub repo)
try:
    model = joblib.load('cardio_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}")

st.title("Cardiovascular Disease Prediction")
st.write("Enter patient details to predict the risk profile.")

col1, col2 = st.columns(2)

with col1:
    age_years = st.number_input("Age (Years)", min_value=1, max_value=120, value=50)
    gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x==1 else "Male")
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165)
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0)
    ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=50, max_value=250, value=120)

with col2:
    ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=30, max_value=150, value=80)
    cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3])
    gluc = st.selectbox("Glucose Level", options=[1, 2, 3])
    smoke = st.checkbox("Smoker")
    alco = st.checkbox("Alcohol Intake")
    active = st.checkbox("Physical Activity", value=True)

if st.button("Predict"):
    # Prepare features
    age_days = age_years * 365.25
    features = np.array([[age_days, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, int(smoke), int(alco), int(active)]])
    
    # Scale features using the loaded scaler
    features_scaled = scaler.transform(features)
    
    # Prediction
    prediction = model.predict(features_scaled)
    prob = model.predict_proba(features_scaled)[0][1]
    
    if prediction[0] == 1:
        st.error(f"High Risk: Prediction Probability {prob:.2%}")
    else:
        st.success(f"Low Risk: Prediction Probability {prob:.2%}")
