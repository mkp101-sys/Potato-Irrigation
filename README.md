# 🥔 Potato Smart Irrigation Advisor

An AI-powered dashboard designed to help farmers optimize irrigation for potato crops. This app uses a **Hybrid LSTM-GRNN model** to predict soil moisture and crop health (NDVI) based on real-time weather data.

## 🚀 Features
* **Real-time Weather Integration:** Connects to OpenWeatherMap API for live temperature and humidity data.
* **AI Predictions:** Predicts soil moisture levels to prevent over-watering or drought stress.
* **Irrigation Alerts:** Provides clear "Action" messages (Irrigate / Safe) based on AI analysis.
* **Growth Tracking:** Calculates Days After Sowing (DAS) to adjust water needs based on the growth stage.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Framework:** Streamlit (Web UI)
* **AI Libraries:** TensorFlow, Keras, Scikit-Learn
* **Data Sources:** OpenWeatherMap API

## 📦 Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone [https://github.com/mkp101-sys/Potato-Irrigation.git](https://github.com/mkp101-sys/Potato-Irrigation.git)
   cd Potato-Irrigation
