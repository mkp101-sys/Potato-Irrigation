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
    # Ensure you upload the .keras version for best compatibility
    model_path = os.path.join(base_path, "potato_hybrid_model.keras")
    scaler_path = os.path.join(base_path, "data_scaler.gz")
    
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = tf.keras.models.load_model(model_path, compile=False)
            scaler = joblib.load(scaler_path)
            return model, scaler
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, scaler = load_assets()

# --- 2. PENMAN-MONTEITH ET0 CALCULATION ---
def calculate_eto_penman(t_max, t_min, t_curr, hum, pres, wind, lat, das):
    """FAO-56 Penman-Monteith Equation"""
    t_avg = (t_max + t_min) / 2
    # Slope of vapor pressure curve
    delta = (4098 * (0.6108 * math.exp(17.27 * t_avg / (t_avg + 237.3)))) / ((t_avg + 237.3)**2)
    # Psychrometric constant (kPa/C)
    psy = 0.000665 * (pres / 10) 
    # Vapor Pressure Deficit
    es_max = 0.6108 * math.exp(17.27 * t_max / (t_max + 237.3))
    es_min = 0.6108 * math.exp(17.27 * t_min / (t_min + 237.3))
    es = (es_max + es_min) / 2
    ea = es * (hum / 100)
    vpd = es - ea
    # Extraterrestrial Radiation (Ra) based on Lat and DAS
    phi = (math.pi / 180) * lat
    dr = 1 + 0.033 * math.cos(2 * math.pi * das / 365)
    sol_dec = 0.409 * math.sin(2 * math.pi * das / 365 - 1.39)
    sha = math.acos(-math.tan(phi) * math.tan(sol_dec))
    ra = (24 * 60 / math.pi) * 0.0820 * dr * (sha * math.sin(phi) * math.sin(sol_dec) + math.cos(phi) * math.cos(sol_dec) * math.sin(sha))
    rn = 0.77 * (0.75 * ra) # Simplified Net Radiation
    
    num = 0.408 * delta * rn + psy * (900 / (t_avg + 273)) * wind * vpd
    den = delta + psy * (1 + 0.34 * wind)
    return max(num / den, 0.1)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Potato AI Advisor", page_icon="🥔")
st.title("🥔 Potato Smart Irrigation Advisor")

# The Three Specific Input Boxes
st.subheader("📋 Step 1: Manual Inputs")
col1, col2, col3 = st.columns(3)

with col1:
    sowing_date = st.date_input("Sowing Date", value=None)
with col2:
    current_date = st.date_input("Current Date", value=None)
with col3:
    # A single box for Lat/Lon (or separate if preferred, here separate for precision)
    lat = st.number_input("Latitude", format="%.4f", value=None)
    lon = st.number_input("Longitude", format="%.4f", value=None)

st.divider()

# --- 4. EXECUTION ---
if st.button("Generate Prediction", use_container_width=True):
    API_KEY = st.secrets.get("OWM_KEY")
    
    if not all([sowing_date, current_date, lat, lon]):
        st.warning("⚠️ Please fill in all three input boxes above.")
    elif not API_KEY:
        st.error("❌ API Key (OWM_KEY) not found in Streamlit Secrets.")
    else:
        # Calculate DAS
        das = (current_date - sowing_date).days
        
        if das < 0:
            st.error("❌ Current Date cannot be earlier than Sowing Date.")
        else:
            # Fetch Weather
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
            try:
                res = requests.get(url).json()
                if res.get("main"):
                    t_curr = res["main"]["temp"]
                    t_max = res["main"]["temp_max"]
                    t_min = res["main"]["temp_min"]
                    hum = res["main"]["humidity"]
                    pres = res["main"]["pressure"]
                    wind = res.get("wind", {}).get("speed", 2.0)

                    # Calculate GDD and ETo
                    gdd = max(0, t_curr - 7) # Potato base temp 7°C
                    eto = calculate_eto_penman(t_max, t_min, t_curr, hum, pres, wind, lat, das)
                    
                    # Determine Stage and Kc
                    if das <= 25: stage, kc = "Vegetative", 0.45
                    elif das <= 70: stage, kc = "Tuber Initiation", 1.15
                    elif das <= 100: stage, kc = "Maturity", 0.85
                    else: stage, kc = "Harvest", 0.70
                    
                    etc = eto * kc # Crop Evapotranspiration

                    # Predict Surface Soil Moisture (SSM)
                    if model and scaler:
                        # Prepare features (Matching your training order)
                        # Assume features are: [DAS, T_curr, T_max, T_min, Hum, Pres, Wind, ETo, GDD]
                        features = np.array([[das, t_curr, t_max, t_min, hum, pres, wind, eto, gdd]])
                        scaled = scaler.transform(features)
                        prediction = model.predict(scaled.reshape(1, 1, 9))
                        ssm_pred = float(prediction[0][0])

                        # DISPLAY RESULTS
                        st.success(f"**Growth Stage:** {stage} (DAS: {das})")
                        
                        r1, r2, r3 = st.columns(3)
                        r1.metric("Ref. ETo", f"{eto:.2f} mm")
                        r2.metric("Crop ETc", f"{etc:.2f} mm")
                        r3.metric("Daily GDD", f"{gdd:.1f}")
                        
                        st.divider()
                        st.metric("Predicted Surface Soil Moisture", f"{ssm_pred:.2f}%")
                        
                        if ssm_pred < 35:
                            st.error(f"🚨 Soil is Dry! Apply {(etc * 1.1):.2f} mm irrigation.")
                        else:
                            st.success("✅ Moisture is sufficient.")
                    else:
                        st.error("Model files not loaded. Check GitHub.")
                else:
                    st.error("City/Weather data not found. Check Lat/Lon.")
            except Exception as e:
                st.error(f"Error fetching data: {e}")

st.divider()
st.caption("Potato Hybrid Advisor | FAO-56 Penman-Monteith Method")
