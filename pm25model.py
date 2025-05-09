import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the scaler and model
scaler = joblib.load('scaler.joblib')
model = joblib.load('model.joblib')

# Function to get health provisions based on PM2.5 level
def get_health_provisions(pm25):
    if pm25 <= 12:
        return ("Good", "Air quality is safe. No specific precautions needed. Enjoy outdoor activities!")
    elif 12.1 <= pm25 <= 35.4:
        return ("Moderate", "Air quality is acceptable. Sensitive groups (e.g., those with respiratory issues) should reduce prolonged outdoor activities.")
    elif 35.5 <= pm25 <= 55.4:
        return ("Unhealthy for Sensitive Groups", "Sensitive groups (children, elderly, those with respiratory conditions) should avoid strenuous outdoor activities. Others should limit prolonged exposure.")
    elif 55.5 <= pm25 <= 150.4:
        return ("Unhealthy", "Everyone should reduce outdoor activities. Sensitive groups should stay indoors and use air purifiers if possible.")
    elif 150.5 <= pm25 <= 250.4:
        return ("Very Unhealthy", "Avoid all outdoor activities. Wear N95 masks if going outside, use air purifiers, and keep windows closed.")
    else:
        return ("Hazardous", "Stay indoors, seal windows, use air purifiers, and wear N95 masks if outdoor exposure is unavoidable. High risk for all.")

# Streamlit app
st.title('PM2.5 Prediction App for Chennai')
st.write('Enter environmental parameters to predict PM2.5 levels and get health recommendations for agricultural areas in Chennai.')

# Input fields for selected features
st.header("Input Parameters")
col1, col2 = st.columns(2)

with col1:
    pm10 = st.number_input('PM10 (µg/m³)', min_value=0.0, value=50.0, step=0.1)
    no2 = st.number_input('NO2 (µg/m³)', min_value=0.0, value=15.0, step=0.1)
    co = st.number_input('CO (mg/m³)', min_value=0.0, value=1.0, step=0.01)
    ozone = st.number_input('Ozone (µg/m³)', min_value=0.0, value=15.0, step=0.1)

with col2:
    at = st.number_input('Ambient Temperature (°C)', min_value=0.0, value=30.0, step=0.1)
    rh = st.number_input('Relative Humidity (%)', min_value=0.0, max_value=100.0, value=75.0, step=0.1)
    ws = st.number_input('Wind Speed (m/s)', min_value=0.0, value=2.0, step=0.1)
    wd = st.number_input('Wind Direction (deg)', min_value=0.0, max_value=360.0, value=180.0, step=1.0)

# Predict button
if st.button('Predict PM2.5'):
    # Create input data array
    input_data = [[pm10, no2, co, ozone, at, rh, ws, wd]]

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Get health provisions
    category, provisions = get_health_provisions(prediction)

    # Display results
    st.success(f'Predicted PM2.5 Level: {prediction:.2f} µg/m³')
    st.subheader(f'Air Quality Category: {category}')
    st.write(f'**Health Provisions for Agricultural Workers**: {provisions}')
    st.info('Note: High PM2.5 levels can affect crop health and worker productivity. Consider protective measures for outdoor farming activities.')

# Additional information
st.sidebar.header("About")
st.sidebar.write("This app predicts PM2.5 levels in Chennai using a Random Forest model trained on environmental data. It provides health recommendations for agricultural workers based on WHO air quality guidelines.")
st.sidebar.write("Features used: PM10, NO2, CO, Ozone, Temperature, Humidity, Wind Speed, Wind Direction.")
