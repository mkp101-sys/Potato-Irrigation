import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import requests
from datetime import datetime

# --- 1. SET UP PAGE CONFIG ---
st.set_page_config(page_title="Potato AI Advisor", page_icon="🥔")

# --- 2. ACCESS API KEY FROM SECRETS ---
# Ensure you have OWM_KEY = "your_key" in the Streamlit Cloud Secrets box
try:
    api_key = st.secrets["OWM_KEY"]
except Exception:
    st.error("Missing API Key! Please add 'OWM_KEY' to your Streamlit Secrets.")
    st.stop()

# --- 3. LOAD AI MODELS (CACHED) ---
@st.cache_resource
def load_assets():
    try:
        # These filenames must match exactly what you upload to GitHub
        model = tf.keras.models.load_model("potato_hybrid_model.h5")
        scaler = joblib.load("data_scaler.gz")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, scaler = load_assets()

# --- 4. USER INTERFACE ---
st.title("🥔 Potato Smart Irrigation Advisor")
st.info("AI-powered moisture prediction for precision farming.")

# Sidebar for inputs
with st.sidebar:
    st.header("Field Settings")
    lat = st.number_input("Latitude (e.g., 23.89)", value=23.8954, format="%.4f")
    lon = st.number_input("Longitude (e.g., 73.12)", value=73.1234, format="%.4f")
    sow_date = st.date_input("Sowing Date", value=datetime(2025, 11, 1))
    st.divider()
    st.write("Current Date:", datetime.now().strftime("%Y-%m-%d"))

# --- 5. PREDICTION LOGIC ---
if st.button("Run AI Analysis", type="primary"):
    if model is not None and scaler is not None:
        with st.spinner("Fetching weather and running AI..."):
            # Fetch Weather from OpenWeatherMap
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            response = requests.get(url).json()
            
            if response.get("cod") == 200:
                tmax = response['main']['temp_max']
                tmin = response['main']['temp_min']
                
                # Calculate Days After Sowing (DAS)
                das = (datetime.now().date() - sow_date).days
                
                # --- AI PREDICTION ---
                # Your model expects 9 features:
                # [T2M_MAX, T2M_MIN, ET0, SSM, NDVI, VH, VV, Daily_GDD, Mech_Baseline]
                # We use local weather + baseline estimates for satellite features
                input_data = np.array([[tmax, tmin, 4.8, 35.0, 0.5, -17.5, -11.5, 14.0, 42.0]])
                
                # Scale and Reshape for LSTM (1 sample, 1 timestep, 9 features)
                scaled_input = scaler.transform(input_data)
                prediction = model.predict(scaled_input.reshape(1, 1, 9), verbose=0)[0][0]
                
                # --- DISPLAY RESULTS ---
                st.subheader(f"Status for Day {das} of Growth")
                
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Predicted Moisture", f"{round(float(prediction), 1)}%")
                kpi2.metric("Max Temp", f"{tmax}°C")
                kpi3.metric("Min Temp", f"{tmin}°C")

                # Decision Logic
                if prediction < 45:
                    st.error("🚨 **IRRIGATE IMMEDIATELY**: Soil moisture is below the safety threshold.")
                elif 45 <= prediction < 60:
                    st.warning("⚠️ **MONITOR CLOSELY**: Soil is drying out. Consider irrigation soon.")
                else:
                    st.success("✅ **SAFE**: Soil moisture is optimal for potato growth.")
            else:
                st.error("Weather API Error. Verify your API Key and Latitude/Longitude.")
    else:
        st.error("Model files not loaded. Ensure .h5 and .gz files are in the GitHub folder.")

# --- 6. FOOTER ---
st.caption("AI Model: Potato Hybrid LSTM-GRNN | Data Source: OpenWeatherMap")
