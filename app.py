import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import requests
from datetime import date

# --- CONFIGURATION ---
try:
    API_KEY = st.secrets["OWM_KEY"]
except:
    API_KEY = "" # Fallback if secret is missing

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        # Using compile=False to fix the Keras/MSE error
        model = tf.keras.models.load_model("potato_hybrid_model.h5", compile=False)
        scaler = joblib.load("data_scaler.gz")
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_assets()

# --- APP UI ---
st.set_page_config(page_title="Potato Irrigation AI", page_icon="🥔")
st.title("🥔 Potato Smart Irrigation Advisor")

# --- INPUT SECTION (The Three Boxes) ---
st.markdown("### 📋 Step 1: Enter Field Details")

col1, col2, col3 = st.columns(3)

with col1:
    # 1. Current Date (No default today)
    current_dt = st.date_input("Current Date", value=None)

with col2:
    # 2. Sowing Date
    sowing_dt = st.date_input("Sowing Date", value=None)

with col3:
    # 3. Location
    location = st.text_input("Location (City)", placeholder="e.g. Dehradun")

# --- WEATHER & PREDICTION ---
if st.button("Calculate & Predict"):
    if not current_dt or not sowing_dt or not location:
        st.warning("Please fill in all three boxes (Current Date, Sowing Date, and Location).")
    else:
        # Calculate DAS (Days After Sowing)
        das = (current_dt - sowing_dt).days
        
        if das < 0:
            st.error("Error: Current date cannot be before Sowing date.")
        else:
            st.info(f"Calculated Days After Sowing (DAS): **{das}**")
            
            # Fetch Weather for the Location entered
            url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
            res = requests.get(url).json()
            
            if res.get("main"):
                temp = res["main"]["temp"]
                hum = res["main"]["humidity"]
                rain = res.get("rain", {}).get("1h", 0)
                
                st.write(f"🌡️ **Weather in {location}:** {temp}°C, Humidity: {hum}%, Rain: {rain}mm")

                if model and scaler:
                    # Prepare data for AI (DAS, Temp, Hum, Rain)
                    input_data = np.array([[das, temp, hum, rain]])
                    input_scaled = scaler.transform(input_data)
                    input_reshaped = input_scaled.reshape((1, 1, 4))
                    
                    # Prediction
                    prediction = model.predict(input_reshaped)
                    moisture = prediction[0][0]
                    
                    st.metric("Predicted Soil Moisture", f"{moisture:.2f}%")
                    
                    if moisture < 30:
                        st.error("🚨 **Action:** Soil is dry. Turn on irrigation!")
                    elif 30 <= moisture <= 60:
                        st.warning("⚠️ **Status:** Moderate moisture.")
                    else:
                        st.success("✅ **Status:** Soil is well hydrated.")
                else:
                    st.error("Model files not found on GitHub.")
            else:
                st.error("Could not find weather data. Please check the City name or API Key.")

st.divider()
st.caption("AI Model: Potato Hybrid LSTM-GRNN | Uses compile=False for compatibility")
