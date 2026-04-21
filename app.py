import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import requests
import os
import math
from datetime import date

# --- 1. SECURE API FETCH ---
try:
    API_KEY = st.secrets["OWM_KEY"]
except Exception:
    API_KEY = None

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "potato_hybrid_model.h5")
    scaler_path = os.path.join(base_path, "data_scaler.gz")
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            # compile=False and safe_mode=False to bypass Keras version mismatch
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            scaler = joblib.load(scaler_path)
            return model, scaler
        return None, None
    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None, None

model, scaler = load_assets()

# --- 3. PENMAN-MONTEITH ETo CALCULATION ---
def calculate_penman_monteith(t_curr, t_max, t_min, hum, pres, wind, lat, das):
    """FAO-56 Penman-Monteith Simplified for Daily Weather Data"""
    # 1. Basics
    t_avg = (t_max + t_min) / 2
    delta = (4098 * (0.6108 * math.exp(17.27 * t_avg / (t_avg + 237.3)))) / ((t_avg + 237.3)**2) # Slope vapor pressure
    psy = 0.000665 * (pres / 10) # Psychrometric constant
    
    # 2. Vapor Pressure Deficit (VPD)
    es_max = 0.6108 * math.exp(17.27 * t_max / (t_max + 237.3))
    es_min = 0.6108 * math.exp(17.27 * t_min / (t_min + 237.3))
    es = (es_max + es_min) / 2
    ea = es * (hum / 100)
    vpd = es - ea
    
    # 3. Net Radiation (Rn) - Simplified Estimate based on Latitude and DAS
    phi = (math.pi / 180) * lat
    dr = 1 + 0.033 * math.cos(2 * math.pi * das / 365)
    sol_dec = 0.409 * math.sin(2 * math.pi * das / 365 - 1.39)
    sha = math.acos(-math.tan(phi) * math.tan(sol_dec))
    ra = (24 * 60 / math.pi) * 0.0820 * dr * (sha * math.sin(phi) * math.sin(sol_dec) + math.cos(phi) * math.cos(sol_dec) * math.sin(sha))
    rn = 0.77 * (0.75 * ra) # Net radiation estimate
    
    # 4. Final Penman-Monteith Equation
    num = 0.408 * delta * rn + psy * (900 / (t_avg + 273)) * wind * vpd
    den = delta + psy * (1 + 0.34 * wind)
    return max(num / den, 0.1)

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Potato Precision AI", page_icon="🥔")
st.title("🥔 Potato Smart Irrigation Advisor")

st.subheader("📋 Manual Input Parameters")
col1, col2 = st.columns(2)

with col1:
    sowing_date = st.date_input("Sowing Date", value=None)
    current_date = st.date_input("Current Date", value=None)

with col2:
    lat = st.number_input("Latitude", format="%.4f", value=None)
    lon = st.number_input("Longitude", format="%.4f", value=None)

# --- 5. PREDICTION ---
if st.button("Generate Prediction", use_container_width=True):
    if not all([sowing_date, current_date, lat, lon]):
        st.warning("⚠️ Please fill in all boxes.")
    elif API_KEY is None:
        st.error("❌ OWM_KEY missing in Secrets.")
    else:
        das = (current_date - sowing_date).days
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        
        try:
            res = requests.get(weather_url).json()
            if res.get("main"):
                t_curr = res["main"]["temp"]
                t_max = res["main"]["temp_max"]
                t_min = res["main"]["temp_min"]
                hum = res["main"]["humidity"]
                pres = res["main"]["pressure"]
                wind = res.get("wind", {}).get("speed", 2.0)
                
                # Calculations
                eto = calculate_penman_monteith(t_curr, t_max, t_min, hum, pres, wind, lat, das)
                gdd = max(0, t_curr - 7)
                
                # Kc and Stage
                if das <= 25: stage, kc = "Vegetative", 0.45
                elif das <= 70: stage, kc = "Tuber Initiation", 1.15
                elif das <= 100: stage, kc = "Maturity", 0.85
                else: stage, kc = "Harvest", 0.70
                
                etc = eto * kc

                if model and scaler:
                    # Construct 9 Features for your model
                    features = np.array([[das, t_curr, t_max, t_min, hum, pres, wind, eto, gdd]])
                    scaled_features = scaler.transform(features)
                    
                    prediction = model.predict(scaled_features.reshape(1, 1, 9))
                    sm_value = float(prediction[0][0])
                    
                    st.divider()
                    st.success(f"📌 Stage: {stage} (DAS: {das})")
                    
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Soil Moisture", f"{sm_value:.1f}%")
                    r2.metric("Root Depth", f"{min(15+(das*0.5), 60):.1f} cm")
                    r3.metric("Water to Apply", f"{(etc * 1.1):.2f} mm")
                else:
                    st.error("Model files not loaded properly.")
            else:
                st.error("Weather data fetch failed.")
        except Exception as e:
            st.error(f"Error: {e}")
