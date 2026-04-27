import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(
    page_title="CardioIntel AI", 
    layout="wide", 
    page_icon="❤️",
    initial_sidebar_state="collapsed"
)

# --- 2. ADVANCED UI STYLING (High Visibility) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }
    
    /* Header Container */
    .main-header {
        background-color: #ffffff;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        text-align: center;
        border-bottom: 5px solid #ff4b4b;
    }
    
    /* Input Card Styling */
    div[data-testid="stForm"] {
        background-color: #ffffff !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 3rem !important;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1) !important;
    }
    
    /* Text Visibility Overrides */
    h1, h2, h3, p, label {
        color: #1e293b !important;
        font-family: 'Inter', sans-serif;
    }
    
    .stNumberInput label, .stSelectbox label {
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        margin-bottom: 10px !important;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff7676 100%);
        color: white !important;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 12px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA LOADING ---
@st.cache_resource
def load_assets():
    try:
        m = joblib.load('cardio_model.pkl')
        s = joblib.load('scaler.pkl')
        return m, s
    except:
        return None, None

model, scaler = load_assets()

# --- 4. TOP DASHBOARD ---
st.markdown("""
    <div class="main-header">
        <h1 style='margin:0; font-size: 2.8rem;'>❤️ CardioIntel Diagnostics</h1>
        <p style='font-size: 1.2rem; color: #64748b;'>AI-Driven Cardiovascular Risk Assessment Engine</p>
    </div>
    """, unsafe_allow_html=True)

if model is None:
    st.error("🚨 System Failure: Predictive models not detected in the repository.")
    st.stop()

# --- 5. INPUT SECTION ---
with st.form("clean_ui_form"):
    t1, t2 = st.columns([1,1])
    with t1:
        st.markdown("### 👤 Patient Identity")
        age = st.number_input("Current Age (Years)", 1, 110, 45)
        gender = st.selectbox("Biological Sex", [1, 2], format_func=lambda x: "Female" if x==1 else "Male")
        height = st.number_input("Height (cm)", 100, 250, 170)
        weight = st.number_input("Body Weight (kg)", 30, 200, 75)

    with t2:
        st.markdown("### 🩺 Clinical Vitals")
        ap_hi = st.number_input("Systolic Blood Pressure", 80, 220, 120)
        ap_lo = st.number_input("Diastolic Blood Pressure", 40, 140, 80)
        chol = st.selectbox("Cholesterol Status", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "High Risk"][x-1])
        gluc = st.selectbox("Glucose Status", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "High Risk"][x-1])

    st.markdown("---")
    l1, l2, l3 = st.columns(3)
    with l1: smoke = st.checkbox("🚭 Active Smoker")
    with l2: alco = st.checkbox("🍷 Regular Alcohol")
    with l3: active = st.checkbox("🏃 Physical Activity", value=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("GENERATE DIAGNOSTIC REPORT")

# --- 6. OUTPUT REPORT ---
if submit:
    # Logic
    input_data = np.array([[age*365.25, gender, height, weight, ap_hi, ap_lo, chol, gluc, int(smoke), int(alco), int(active)]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    prob = model.predict_proba(scaled_data)[0][1]

    st.markdown("## 📊 Diagnostic Result")
    
    r1, r2 = st.columns([1, 1.5])
    
    with r1:
        if prediction[0] == 1:
            st.markdown(f"""
                <div style="background-color: #fff1f0; padding: 2rem; border-radius: 20px; border: 2px solid #ffa39e; text-align: center;">
                    <h2 style="color: #cf1322 !important;">HIGH RISK</h2>
                    <h1 style="font-size: 4rem; color: #cf1322 !important;">{prob:.1%}</h1>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color: #f6ffed; padding: 2rem; border-radius: 20px; border: 2px solid #b7eb8f; text-align: center;">
                    <h2 style="color: #389e0d !important;">LOW RISK</h2>
                    <h1 style="font-size: 4rem; color: #389e0d !important;">{prob:.1%}</h1>
                </div>
            """, unsafe_allow_html=True)

    with r2:
        st.write("### AI Risk Breakdown")
        st.progress(prob)
        
        # Smart Advice based on inputs
        if ap_hi >= 140:
            st.warning("⚠️ **Hypertension detected.** Blood pressure is a primary driver for your result.")
        if chol > 1:
            st.info("ℹ️ **Lipid Profile:** Elevated cholesterol levels are increasing arterial stress.")
        if not active:
            st.write("💡 **Recommendation:** Daily physical activity could lower this risk score significantly.")
