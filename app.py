import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import requests
import os
import math
from datetime import date

# --- 1. SECURE API FETCH ---
# This pulls the key you saved in Streamlit's "Secrets" box
try:
    API_KEY = st.secrets["OWM_KEY"]
except Exception:
    st.error("Missing API Key! Please add 'OWM_KEY' to Streamlit Secrets.")
    API_KEY = None

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "potato_hybrid_model.h5")
    scaler_path = os.path.join(base_path, "data_scaler.gz")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        # compile=False handles the Keras version mismatch error
        model = tf.keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

model, scaler = load_assets()

# --- 3. AGRONOMIC CALCULATIONS ---
def get_potato_stage(das):
    if das <= 25: return "Sprouting/Vegetative", 0.45
    elif das <= 70: return "Tuber Initiation/Bulking", 1.15
    elif das <= 100: return "Maturity", 0.85
    else: return "Late Season/Harvest", 0.70

def estimate_eto(t_max, t_min, rh):
    # Simplified Penman-Monteith (Hargreaves variant)
    t_avg = (t_max + t_min) / 2
    eto = 0.0023 * (t_avg + 17.8) * math.sqrt(t_max - t_min) * 0.4 * 15 
    return max(eto, 0.1)

# --- 4. APP INTERFACE ---
st.set_page_config(page_title="Potato AI Advisor", page_icon="🥔")
st.title("🥔 Potato Smart Irrigation Advisor")

# Input Boxes (No Defaults)
st.subheader("Field Parameters")
col1, col2 = st.columns(2)

with col1:
    sowing_date = st.date_input("Sowing Date", value=None)
    current_date = st.date_input("Current Date", value=None)

with col2:
    lat = st.number_input("Latitude", format="%.4f", value=None)
    lon = st.number_input("Longitude", format="%.4f", value=None)

st.divider()

# --- 5. EXECUTION ---
if st.button("Generate Prediction", use_container_width=True):
    if not all([sowing_date, current_date, lat, lon, API_KEY]):
        st.warning("Please fill all inputs and ensure API Key is set in Secrets.")
    else:
        # Calculate DAS
        das = (current_date - sowing_date).days
        
        if das < 0:
            st.error("Error: Current date cannot be before Sowing date.")
        else:
            # Fetch Weather
            weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
            data = requests.get(weather_url).json()

            if data.get("main"):
                t_curr = data["main"]["temp"]
                t_max = data["main"]["temp_max"]
                t_min = data["main"]["temp_min"]
                hum = data["main"]["humidity"]
                
                # Agronomic Logic
                stage, kc = get_potato_stage(das)
                gdd = max(0, t_curr - 7) # Base Temp 7°C
                eto = estimate_eto(t_max, t_min, hum)
                etc = eto * kc
                
                st.success(f"**Growth Stage:** {stage} (DAS: {das})")
                
                # AI Model Prediction
                if model and scaler:
                    # Prepare features (ensure this order matches your training)
                    # Example: [DAS, Temp, Humidity, ETo]
                    features = np.array([[das, t_curr, hum, eto]])
                    scaled_features = scaler.transform(features)
                    
                    prediction = model.predict(scaled_features.reshape(1, 1, 4))
                    sm_result = float(prediction[0][0])
                    
                    # Calculated Outputs
                    root_depth = min(15 + (das * 0.5), 60) # Root growth estimate in cm
                    water_applied = etc * 1.1 # Example 10% leaching factor
                    
                    # Display Results
                    res1, res2, res3 = st.columns(3)
                    res1.metric("Soil Moisture", f"{sm_result:.1f}%")
                    res2.metric("Root Depth", f"{root_depth:.1f} cm")
                    res3.metric("Water to Apply", f"{water_applied:.2f} mm")
                    
                    if sm_result < 35:
                        st.error("⚠️ Status: Critical. Apply irrigation now.")
                    else:
                        st.success("✅ Status: Optimal. No irrigation needed today.")
                else:
                    st.error("Model files (.h5 / .gz) not detected in GitHub.")
            else:
                st.error("Weather data fetch failed. Verify Latitude/Longitude.")

st.divider()
st.caption("Secured with Streamlit Secrets | Hybrid LSTM-GRNN Model")
