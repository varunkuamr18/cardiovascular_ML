import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="CardioCheck Pro", 
    layout="wide", 
    page_icon="🧬"
)

# --- 2. HIGH-CONTRAST DARK UI DESIGN ---
st.markdown("""
    <style>
    /* Background and Global Text */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Custom Card for Inputs */
    div[data-testid="stForm"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 20px !important;
        padding: 2.5rem !important;
    }

    /* Labels and Headers Visibility */
    label, p, h1, h2, h3 {
        color: #e6edf3 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stNumberInput label, .stSelectbox label {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #58a6ff !important; /* Neon Blue for labels */
    }

    /* Input Fields */
    input {
        background-color: #0d1117 !important;
        color: white !important;
        border: 1px solid #30363d !important;
    }

    /* Professional Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1f6feb 0%, #58a6ff 100%);
        color: white !important;
        border: none;
        padding: 0.8rem;
        font-size: 1.3rem;
        font-weight: bold;
        border-radius: 10px;
        margin-top: 20px;
    }
    
    /* Result Boxes */
    .high-risk-box {
        background-color: #3e1e1e;
        border: 2px solid #f85149;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    .low-risk-box {
        background-color: #1b2e1b;
        border: 2px solid #3fb950;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        m = joblib.load('cardio_model.pkl')
        s = joblib.load('scaler.pkl')
        return m, s
    except:
        return None, None

model, scaler = load_assets()

# --- 4. HEADER ---
st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🧬 Cardiovascular Analysis Portal</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>High-precision AI diagnostics for clinical assessment</p>", unsafe_allow_html=True)

if model is None:
    st.error("System Error: model/scaler files not found.")
    st.stop()

# --- 5. INPUT FORM ---
with st.form("dark_ui_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='color: #8b949e;'>Patient Metrics</h3>", unsafe_allow_html=True)
        age = st.number_input("Age (Years)", 1, 110, 45)
        height = st.number_input("Height (cm)", 100, 250, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 75)
        gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x==1 else "Male")

    with col2:
        st.markdown("<h3 style='color: #8b949e;'>Vitals & Labs</h3>", unsafe_allow_html=True)
        ap_hi = st.number_input("Systolic BP", 80, 220, 120)
        ap_lo = st.number_input("Diastolic BP", 40, 140, 80)
        chol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: ["Normal", "Borderline", "High"][x-1])
        gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: ["Normal", "Borderline", "High"][x-1])

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: smoke = st.checkbox("🚬 Smoker")
    with c2: alco = st.checkbox("🍺 Alcohol")
    with c3: active = st.checkbox("🏃 Active", value=True)
    
    submit = st.form_submit_button("RUN DIAGNOSTIC")

# --- 6. RESULTS ---
if submit:
    input_data = np.array([[age*365.25, gender, height, weight, ap_hi, ap_lo, chol, gluc, int(smoke), int(alco), int(active)]])
    scaled_data = scaler.transform(input_data)
    prob = model.predict_proba(scaled_data)[0][1]
    prediction = model.predict(scaled_data)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if prediction[0] == 1:
        st.markdown(f"""
            <div class="high-risk-box">
                <h2 style="color: #f85149 !important; margin: 0;">POSITIVE: HIGH RISK</h2>
                <h1 style="color: #f85149 !important; font-size: 3.5rem;">{prob:.1%}</h1>
                <p style="color: #f85149 !important;">Immediate clinical consultation recommended.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="low-risk-box">
                <h2 style="color: #3fb950 !important; margin: 0;">NEGATIVE: LOW RISK</h2>
                <h1 style="color: #3fb950 !important; font-size: 3.5rem;">{prob:.1%}</h1>
                <p style="color: #3fb950 !important;">Patient metrics are within acceptable limits.</p>
            </div>
        """, unsafe_allow_html=True)
