import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
try:
    model = pickle.load(open('cardio_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except:
    st.error("Model files not found. Please run the training script first.")

st.set_page_config(page_title="CardioCheck AI", layout="wide")

st.title("🏥 Cardiovascular Disease Prediction Portal")
st.markdown("Enter the patient's clinical data to calculate the probability of cardiovascular disease.")

# Create Input Form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (Years)", 1, 120, 50)
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x==1 else "Male")
        height = st.number_input("Height (cm)", 50, 250, 165)
        
    with col2:
        weight = st.number_input("Weight (kg)", 10.0, 300.0, 70.0)
        ap_hi = st.number_input("Systolic BP (ap_hi)", 50, 250, 120)
        ap_lo = st.number_input("Diastolic BP (ap_lo)", 30, 150, 80)
        
    with col3:
        chol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
        gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
        smoke = st.toggle("Smoker")
        alco = st.toggle("Alcohol Intake")
        active = st.toggle("Physically Active", value=True)

    submit = st.form_submit_button("Generate Report")

if submit:
    # Prepare data (Convert age to days for model compatibility)
    input_data = np.array([[age*365, gender, height, weight, ap_hi, ap_lo, chol, gluc, int(smoke), int(alco), int(active)]])
    
    # Scale inputs
    scaled_data = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(scaled_data)
    prob = model.predict_proba(scaled_data)[0][1]

    # Result Display
    st.divider()
    if prediction[0] == 1:
        st.error(f"### High Risk Detected")
        st.write(f"The model predicts a **{prob*100:.1f}%** probability of Cardiovascular disease.")
    else:
        st.success(f"### Low Risk Profile")
        st.write(f"The model predicts a **{prob*100:.1f}%** probability of Cardiovascular disease.")
    
    # Display Z-Score / Metrics Info
    st.info("Note: This model was trained with GridSearchCV using Random Forest for maximum efficiency.")
