import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import requests
from datetime import datetime

# --- CONFIGURATION ---
# Replace with your actual OpenWeatherMap API Key in Streamlit Secrets
try:
    API_KEY = st.secrets["OWM_KEY"]
except:
    API_KEY = "YOUR_KEY_HERE" 

CITY = "Dehradun" # You can change this to your location

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        # The 'compile=False' fix is included here
        model = tf.keras.models.load_model("potato_hybrid_model.h5", compile=False)
        scaler = joblib.load("data_scaler.gz")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, scaler = load_assets()

# --- APP UI ---
st.set_page_config(page_title="Potato Irrigation AI", page_icon="🥔")

st.title("🥔 Potato Smart Irrigation Advisor")
st.markdown("### AI-powered moisture prediction for precision farming")

# Display Current Date
current_time = datetime.now().strftime("%B %d, %Y | %H:%M")
st.info(f"📅 **Current Date & Time:** {current_time}")

# --- INPUT SECTION ---
st.sidebar.header("Field Data")
das = st.sidebar.slider("Days After Sowing (DAS)", 1, 120, 30)

# Fetch Weather Data
def get_weather(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    res = requests.get(url).json()
    if res.get("main"):
        return {
            "temp": res["main"]["temp"],
            "hum": res["main"]["humidity"],
            "rain": res.get("rain", {}).get("1h", 0)
        }
    return None

weather = get_weather(API_KEY, CITY)

if weather:
    st.sidebar.subheader(f"Weather in {CITY}")
    temp = st.sidebar.number_input("Temperature (°C)", value=float(weather['temp']))
    hum = st.sidebar.number_input("Humidity (%)", value=float(weather['hum']))
    rain = st.sidebar.number_input("Rainfall (mm)", value=float(weather['rain']))
else:
    st.sidebar.warning("Weather API not connected. Please enter manually.")
    temp = st.sidebar.number_input("Temperature (°C)", value=25.0)
    hum = st.sidebar.number_input("Humidity (%)", value=60.0)
    rain = st.sidebar.number_input("Rainfall (mm)", value=0.0)

# --- PREDICTION LOGIC ---
if st.button("Predict Soil Moisture"):
    if model and scaler:
        # Prepare input for prediction
        # (Assuming your model expects: DAS, Temp, Hum, Rain)
        input_data = np.array([[das, temp, hum, rain]])
        input_scaled = scaler.transform(input_data)
        
        # Reshape for LSTM/GRNN (1 sample, 1 time step, 4 features)
        input_reshaped = input_scaled.reshape((1, 1, 4))
        
        prediction = model.predict(input_reshaped)
        moisture = prediction[0][0]
        
        st.metric("Predicted Soil Moisture", f"{moisture:.2f}%")
        
        if moisture < 30:
            st.error("🚨 **Action Required:** Soil is too dry. Turn on irrigation!")
        elif 30 <= moisture <= 60:
            st.warning("⚠️ **Watchful:** Soil moisture is moderate.")
        else:
            st.success("✅ **Status:** Soil moisture is optimal. No irrigation needed.")
    else:
        st.error("Model files are missing. Please check your GitHub repository.")

st.divider()
st.caption("AI Model: Potato Hybrid LSTM-GRNN | Data Source: OpenWeatherMap")
