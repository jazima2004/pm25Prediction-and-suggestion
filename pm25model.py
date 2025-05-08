# Install required libraries
!pip install streamlit pyngrok -q

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load and preprocess dataset
df = pd.read_csv('City_wise_raw_data_1Day_2024_Chennai_1Day.csv')
features = ['PM10 (µg/m³)', 'NO2 (µg/m³)', 'CO (mg/m³)', 'Ozone (µg/m³)', 'AT (°C)', 'RH (%)', 'WS (m/s)', 'WD (deg)']
X = df[features].fillna(df[features].mean())
y = df['PM2.5 (µg/m³)'].fillna(df['PM2.5 (µg/m³)'].mean())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save scaler and model
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(model, 'model.joblib')

# Write app.py (from the artifact above)
with open('app.py', 'w') as f:
    f.write('''
import streamlit as st
import pandas as pd
import joblib
import time

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stNumberInput input {
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 5px;
    }
    .stSlider label {
        font-weight: bold;
        color: #333;
    }
    .sidebar .sidebar-content {
        background-color: #e6f3ff;
        border-radius: 10px;
        padding: 15px;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Load the scaler and model
scaler = joblib.load('scaler.joblib')
model = joblib.load('model.joblib')

# Function to get health provisions based on PM2.5 level
def get_health_provisions(pm25):
    if pm25 <= 12:
        return ("Good", "Air quality is safe. No specific precautions needed. Enjoy outdoor activities!", "#00e676")
    elif 12.1 <= pm25 <= 35.4:
        return ("Moderate", "Air quality is acceptable. Sensitive groups (e.g., those with respiratory issues) should reduce prolonged outdoor activities.", "#ffeb3b")
    elif 35.5 <= pm25 <= 55.4:
        return ("Unhealthy for Sensitive Groups", "Sensitive groups (children, elderly, those with respiratory conditions) should avoid strenuous outdoor activities. Others should limit prolonged exposure.", "#ff9800")
    elif 55.5 <= pm25 <= 150.4:
        return ("Unhealthy", "Everyone should reduce outdoor activities. Sensitive groups should stay indoors and use air purifiers if possible.", "#f44336")
    elif 150.5 <= pm25 <= 250.4:
        return ("Very Unhealthy", "Avoid all outdoor activities. Wear N95 masks if going outside, use air purifiers, and keep windows closed.", "#9c27b0")
    else:
        return ("Hazardous", "Stay indoors, seal windows, use air purifiers, and wear N95 masks if outdoor exposure is unavoidable. High risk for all.", "#3f51b5")

# Sidebar content
with st.sidebar:
    st.header("About the App")
    st.markdown("""
    This app predicts **PM2.5 levels** in Chennai using a machine learning model trained on 2024 air quality data. Enter environmental parameters to get a PM2.5 prediction and tailored **health recommendations**.
    
    ### Air Quality Categories
    - **Good (0–12 µg/m³)**: Safe for all.
    - **Moderate (12.1–35.4 µg/m³)**: Caution for sensitive groups.
    - **Unhealthy for Sensitive Groups (35.5–55.4 µg/m³)**: Avoid strenuous activities.
    - **Unhealthy (55.5–150.4 µg/m³)**: Reduce outdoor exposure.
    - **Very Unhealthy (150.5–250.4 µg/m³)**: Stay indoors, use N95 masks.
    - **Hazardous (>250.4 µg/m³)**: High risk, remain indoors.
    """)
    st.markdown("Developed by [Your Name] | Powered by Streamlit")

# Main app content
st.title('🌬️ Chennai PM2.5 Prediction App')
st.markdown("**Predict PM2.5 levels and get health recommendations based on air quality and weather data.**")

# Input form in a container
with st.container():
    st.subheader("📏 Enter Environmental Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        pm10 = st.slider('PM10 (µg/m³)', min_value=0.0, max_value=200.0, value=50.0, step=0.1)
        no2 = st.slider('NO2 (µg/m³)', min_value=0.0, max_value=100.0, value=15.0, step=0.1)
        co = st.slider('CO (mg/m³)', min_value=0.0, max_value=10.0, value=1.0, step=0.01)
        ozone = st.slider('Ozone (µg/m³)', min_value=0.0, max_value=100.0, value=15.0, step=0.1)
    
    with col2:
        at = st.slider('Ambient Temperature (°C)', min_value=0.0, max_value=50.0, value=30.0, step=0.1)
        rh = st.slider('Relative Humidity (%)', min_value=0.0, max_value=100.0, value=75.0, step=0.1)
        ws = st.slider('Wind Speed (m/s)', min_value=0.0, max_value=20.0, value=2.0, step=0.1)
        wd = st.slider('Wind Direction (deg)', min_value=0.0, max_value=360.0, value=180.0, step=1.0)

# Predict button
if st.button('🔍 Predict PM2.5 Level'):
    # Progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    
    # Create input data array
    input_data = [[pm10, no2, co, ozone, at, rh, ws, wd]]
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Get health provisions
    category, provisions, color = get_health_provisions(prediction)
    
    # Display results in a styled container
    with st.container():
        st.markdown(f"### Predicted PM2.5 Level: **{prediction:.2f} µg/m³**")
        st.markdown(f"#### Air Quality Category: <span style='color:{color};'>{category}</span>", unsafe_allow_html=True)
        st.markdown(f"**Health Recommendations**: {provisions}")
        
        # Visual indicator
        st.markdown(f"<div style='background-color:{color};height:10px;border-radius:5px;'></div>", unsafe_allow_html=True)
    
    progress_bar.empty()
''')

# Create requirements.txt
with open('requirements.txt', 'w') as f:
    f.write('''
streamlit==1.35.0
pandas==2.0.3
numpy==1.25.2
scikit-learn==1.3.0
joblib==1.4.2
''')

# Create .gitignore
with open('.gitignore', 'w') as f:
    f.write('''
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
ENV/
.env
# Dataset (excluded due to size)
City_wise_raw_data_1Day_2024_Chennai_1Day.csv
# Colab-specific
*.ipynb
*.ipynb_checkpoints/
# Misc
.DS_Store
*.log
''')

# Create README.md
with open('README.md', 'w') as f:
    f.write('''
# Chennai PM2.5 Prediction App
This Streamlit app predicts PM2.5 levels in Chennai using a Random Forest Regressor model trained on 2024 air quality data. It provides health recommendations based on the predicted PM2.5 levels, with an enhanced user interface for better engagement.

## Project Structure
- `app.py`: Streamlit app with styled UI.
- `scaler.joblib`: Saved StandardScaler.
- `model.joblib`: Saved Random Forest model.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Excludes unnecessary files.
- `README.md`: Project documentation.

## Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deployment
- Deploy on [Streamlit Cloud](https://streamlit.io/cloud) by linking this GitHub repo.
- Ensure `scaler.joblib` and `model.joblib` are in the repo.

## Notes
- The dataset (`City_wise_raw_data_1Day_2024_Chennai_1Day.csv`) is not included due to GitHub size limits. Host it externally (e.g., Google Drive) and update `app.py` if needed.
- Replace `[Your Name]` in `app.py` and `README.md` with your name.

## License
MIT License
''')

print("Project files generated: app.py, requirements.txt, .gitignore, README.md, scaler.joblib, model.joblib")
print("Follow GitHub upload instructions below.")
