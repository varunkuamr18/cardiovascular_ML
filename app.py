import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# --- 1. PAGE CONFIG & CUSTOM STYLING ---
st.set_page_config(page_title="CardioCheck AI", layout="wide", page_icon="🏥")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stForm {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .report-card {
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 10px solid;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_model_assets():
    try:
        m = joblib.load('cardio_model.pkl')
        s = joblib.load('scaler.pkl')
        return m, s
    except Exception as e:
        return None, None

model, scaler = load_model_assets()

# --- 3. SIDEBAR DESIGN ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=100)
    st.title("About CardioCheck")
    st.info("This AI tool uses Random Forest classification to predict cardiovascular risk based on clinical parameters.")
    st.warning("Note: This is for educational purposes and not a substitute for professional medical advice.")

# --- 4. MAIN UI ---
st.title("🏥 Cardiovascular Health Intelligence")
st.markdown("---")

if model is None or scaler is None:
    st.error("⚠️ System Offline: Model assets not found in repository.")
    st.stop()

# --- 5. INPUT FORM ---
with st.form("prediction_form"):
    st.subheader("Patient Clinical Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Personal Data**")
        age = st.number_input("Age (Years)", 1, 120, 50)
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x==1 else "Male")
        height = st.number_input("Height (cm)", 50, 250, 165)
        weight = st.number_input("Weight (kg)", 20.0, 300.0, 70.0)
    
    with col2:
        st.write("**Vitals**")
        ap_hi = st.number_input("Systolic BP", 50, 250, 120)
        ap_lo = st.number_input("Diastolic BP", 30, 150, 80)
        active = st.toggle("Physically Active", value=True)
        smoke = st.toggle("Smoker")
        alco = st.toggle("Alcohol Intake")

    with col3:
        st.write("**Lab Results**")
        chol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])
        gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "High"][x-1])

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("ANALYSIS PATIENT RISK")

# --- 6. PREDICTION & ENHANCED REPORT ---
if submit:
    # Prepare and Scale data
    input_data = np.array([[age*365.25, gender, height, weight, ap_hi, ap_lo, chol, gluc, int(smoke), int(alco), int(active)]])
    scaled_data = scaler.transform(input_data)
    
    # Prediction
    prediction = model.predict(scaled_data)
    prob = model.predict_proba(scaled_data)[0][1]

    st.subheader("Diagnostic Summary")
    
    # Design Result Cards
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction[0] == 1:
            st.markdown(f"""
                <div class='report-card' style='background-color: #ffe5e5; border-color: #ff4b4b; color: #ff4b4b;'>
                    <h3>HIGH RISK</h3>
                    <h1>{prob:.1%}</h1>
                    <p>Cardiovascular risk detected.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='report-card' style='background-color: #e5f9e5; border-color: #28a745; color: #28a745;'>
                    <h3>LOW RISK</h3>
                    <h1>{prob:.1%}</h1>
                    <p>Within safe parameters.</p>
                </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.write("**Risk Analysis & Suggestions:**")
        if ap_hi > 140 or ap_lo > 90:
            st.write("🔴 **Hypertension Alert:** Blood pressure is outside normal range.")
        if chol > 1:
            st.write("🟡 **Cholesterol Warning:** Elevated cholesterol contributes to arterial plaque.")
        if not active:
            st.write("🔵 **Activity Recommendation:** Increase physical activity to improve heart health.")
        
        st.progress(prob) # Visual probability bar
