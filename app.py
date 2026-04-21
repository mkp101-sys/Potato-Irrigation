import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import requests
import os
import math
from datetime import date

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    # Pointing to the new .keras format for better compatibility
    model_path = os.path.join(base_path, "potato_hybrid_model.keras")
    scaler_path = os.path.join(base_path, "data_scaler.gz")
    
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        return None, None
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None, None

model, scaler = load_assets()

# --- 2. PENMAN-MONTEITH ET0 FUNCTION ---
def calculate_eto_penman(t_max, t_min, t_curr, hum, pres, wind, lat, das):
    """Full FAO-56 Penman-Monteith Equation"""
    t_avg = (t_max + t_min) / 2
    
    # 1. Slope of vapor pressure curve (Delta)
    delta = (4098 * (0.6108 * math.exp(17.27 * t_avg / (t_avg + 237.3)))) / ((t_avg + 237.3)**2)
    
    # 2. Psychrometric constant (gamma)
    psy = 0.000665 * (pres / 10) # Pressure in kPa
    
    # 3. Vapor Pressure Deficit (VPD)
    es_max = 0.6108 * math.exp(17.27 * t_max / (t_max + 237.3))
    es_min = 0.6108 * math.exp(17.27 * t_min / (t_min + 237.3))
    es = (es_max + es_min) / 2
    ea = es * (hum / 100)
    vpd = es - ea
    
    # 4. Net Radiation (Rn) - Simplified for daily step
    dr = 1 + 0.033 * math.cos(2 * math.pi * das / 365)
    sol_dec = 0.409 * math.sin(2 * math.pi * das / 365 - 1.39)
    phi = (math.pi / 180) * lat
    sha = math.acos(-math.tan(phi) * math.tan(sol_dec))
    ra = (24 * 60 / math.pi) * 0.0820 * dr * (sha * math.sin(phi) * math.sin(sol_dec) + math.cos(phi) * math.cos(sol_dec) * math.sin(sha))
    rn = 0.77 * (0.75 * ra) # Net radiation estimate (0.77 is albedo for grass)
    
    # 5. Final ET0
    num = 0.408 * delta * rn + psy * (900 / (t_avg + 273)) * wind * vpd
    den = delta + psy * (1 + 0.34 * wind)
    return max(num / den, 0.1)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Potato ETc Predictor", page_icon="🥔")
st.title("🥔 Potato Precision Irrigation Advisor")
st.markdown("Calculates **ET₀ (Penman-Monteith)** → **ET꜀ (Crop ET)** → **Soil Moisture Prediction**")

# Input Section
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        sowing_date = st.date_input("Sowing Date", value=None)
        current_date = st.date_input("Current Date", value=None)
    with col2:
        lat = st.number_input("Latitude", format="%.4f", value=None)
        lon = st.number_input("Longitude", format="%.4f", value=None)

st.markdown("### 📡 Remote Sensing & Baseline Inputs")
c1, c2, c3 = st.columns(3)
with c1:
    ndvi = st.number_input("NDVI Mean", value=0.50)
    vh = st.number_input("VH Mean", value=-15.0)
with c2:
    vv = st.number_input("VV Mean", value=-10.0)
    ssm = st.number_input("SSM % (Surface)", value=20.0)
with c3:
    baseline = st.number_input("Mech. Baseline (from previous day)", value=25.0)

# --- 4. PREDICTION LOGIC ---
if st.button("Calculate ETc & Predict Moisture", use_container_width=True):
    API_KEY = st.secrets.get("OWM_KEY")
    
    if not all([sowing_date, current_date, lat, lon]):
        st.warning("⚠️ Please fill in all primary field boxes.")
    elif not API_KEY:
        st.error("❌ API Key missing in Secrets.")
    else:
        das = (current_date - sowing_date).days
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        
        try:
            res = requests.get(url).json()
            if res.get("main"):
                # Weather variables
                t_curr = res["main"]["temp"]
                t_max = res["main"]["temp_max"]
                t_min = res["main"]["temp_min"]
                hum = res["main"]["humidity"]
                pres = res["main"]["pressure"]
                wind = res.get("wind", {}).get("speed", 2.0)
                
                # 1. Calculate ET0 (Penman-Monteith)
                eto = calculate_eto_penman(t_max, t_min, t_curr, hum, pres, wind, lat, das)
                
                # 2. Get Kc (Crop Coefficient) and Stage
                if das <= 25: stage, kc = "Vegetative", 0.45
                elif das <= 70: stage, kc = "Tuber Initiation", 1.15
                elif das <= 100: stage, kc = "Maturity", 0.85
                else: stage, kc = "Harvest", 0.70
                
                # 3. Calculate ETc (Crop Evapotranspiration)
                etc = eto * kc
                gdd = max(0, t_curr - 7)

                if model and scaler:
                    # 4. Prepare 9-Feature vector for AI
                    # ['T2M_MAX', 'T2M_MIN', 'ET0', 'SSM', 'NDVI', 'VH', 'VV', 'GDD', 'Mech']
                    input_data = np.array([[t_max, t_min, eto, ssm, ndvi, vh, vv, gdd, baseline]])
                    scaled = scaler.transform(input_data)
                    prediction = model.predict(scaled.reshape(1, 1, 9))
                    rzsm = float(prediction[0][0])
                    
                    # UI Display
                    st.divider()
                    st.success(f"**Growth Stage:** {stage} | **DAS:** {das}")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("ET₀ (Reference)", f"{eto:.2f} mm")
                    m2.metric("Kc (Coefficient)", f"{kc}")
                    m3.metric("ET꜀ (Crop ET)", f"{etc:.2f} mm")
                    
                    st.divider()
                    st.metric("Predicted Root Zone Moisture", f"{rzsm:.2f}%")
                    
                    if rzsm < 35:
                        st.error(f"🚨 Irrigation Needed! Apply approx {(etc * 1.2):.2f} mm of water.")
                    else:
                        st.success("✅ Moisture level is stable.")
            else:
                st.error("Weather service unavailable. Check Lat/Lon.")
        except Exception as e:
            st.error(f"Error: {e}")
