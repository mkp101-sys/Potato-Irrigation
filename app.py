import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import requests
import os
import math
from datetime import date

# --- 1. SECURE API FETCH ---
# Pulls the key from Streamlit's "Secrets" (Advanced Settings)
try:
    API_KEY = st.secrets["OWM_KEY"]
except Exception:
    API_KEY = None

# --- 2. ASSET LOADING (With Fix for Custom Layers) ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "potato_hybrid_model.h5")
    scaler_path = os.path.join(base_path, "data_scaler.gz")
    
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            # compile=False and safe_mode=False are CRITICAL for custom Hybrid models
            model = tf.keras.models.load_model(
                model_path, 
                compile=False, 
                safe_mode=False
            )
            scaler = joblib.load(scaler_path)
            return model, scaler
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, scaler = load_assets()

# --- 3. POTATO LOGIC FUNCTIONS ---
def get_potato_stage(das):
    """Returns growth stage and Crop Coefficient (Kc)"""
    if das <= 25: return "Sprouting/Vegetative", 0.45
    elif das <= 70: return "Tuber Initiation/Bulking", 1.15
    elif das <= 100: return "Maturity", 0.85
    else: return "Late Season/Harvest", 0.70

def calculate_eto(t_max, t_min, rh):
    """Calculates Reference Evapotranspiration (ETo) - Hargreaves Variant"""
    t_avg = (t_max + t_min) / 2
    # Standard formula for daily ETo estimate
    eto = 0.0023 * (t_avg + 17.8) * math.sqrt(abs(t_max - t_min)) * 0.4 * 15 
    return max(eto, 0.1)

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Potato Precision AI", page_icon="🥔")
st.title("🥔 Potato Smart Irrigation Advisor")
st.markdown("---")

# Input Boxes (Starting Empty)
st.subheader("📋 Input Parameters")
col1, col2 = st.columns(2)

with col1:
    sowing_date = st.date_input("Sowing Date", value=None)
    current_date = st.date_input("Current Date", value=None)

with col2:
    lat = st.number_input("Latitude", format="%.4f", value=None)
    lon = st.number_input("Longitude", format="%.4f", value=None)

st.markdown("---")

# --- 5. CALCULATION & PREDICTION ---
if st.button("Generate Prediction", use_container_width=True):
    if not all([sowing_date, current_date, lat, lon]):
        st.warning("⚠️ Please fill in all input boxes above.")
    elif API_KEY is None:
        st.error("❌ API Key not found in Streamlit Secrets!")
    else:
        # Calculate DAS
        das = (current_date - sowing_date).days
        
        if das < 0:
            st.error("❌ Current Date cannot be earlier than Sowing Date.")
        else:
            # Fetch Weather using Lat/Lon
            weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
            try:
                response = requests.get(weather_url)
                data = response.json()
                
                if data.get("main"):
                    t_curr = data["main"]["temp"]
                    t_max = data["main"]["temp_max"]
                    t_min = data["main"]["temp_min"]
                    hum = data["main"]["humidity"]
                    
                    # Compute Agronomic values
                    stage, kc = get_potato_stage(das)
                    gdd = max(0, t_curr - 7) # Base Temp 7°C for Potatoes
                    eto = calculate_eto(t_max, t_min, hum)
                    etc = eto * kc # Crop Evapotranspiration
                    
                    st.success(f"📌 **Stage:** {stage} (DAS: {das})")
                    st.info(f"📊 **Calculated:** GDD: {gdd:.1f} | ETo: {eto:.2f} mm/day | Kc: {kc}")

                    # Run AI Prediction
                    if model and scaler:
                        # Prepare features: DAS, Temp, Humidity, ETo
                        features = np.array([[das, t_curr, hum, eto]])
                        scaled_features = scaler.transform(features)
                        
                        # Reshape for LSTM: (Batch, TimeSteps, Features)
                        prediction = model.predict(scaled_features.reshape(1, 1, 4))
                        sm_value = float(prediction[0][0])
                        
                        # Irrigation Math
                        root_depth = min(15 + (das * 0.5), 60) # Dynamic root zone
                        water_needed = max(0, etc * 1.1)       # 10% leaching factor
                        
                        # Display Results
                        st.divider()
                        r1, r2, r3 = st.columns(3)
                        r1.metric("Soil Moisture", f"{sm_value:.1f}%")
                        r2.metric("Root Depth", f"{root_depth:.1f} cm")
                        r3.metric("Irrigation Volume", f"{water_needed:.2f} mm")
                        
                        if sm_value < 35:
                            st.error("🚨 **Alert:** Soil moisture is low. Irrigate now!")
                        else:
                            st.success("✅ **Status:** Soil moisture is sufficient.")
                    else:
                        st.error("❌ Model or Scaler file not found. Check GitHub folder.")
                else:
                    st.error("❌ Location not recognized. Check Latitude/Longitude.")
            except Exception as err:
                st.error(f"❌ Connection error: {err}")

st.divider()
st.caption("AI Engine: Hybrid LSTM-GRNN | Penman-Monteith Math | Streamlit Secure")
