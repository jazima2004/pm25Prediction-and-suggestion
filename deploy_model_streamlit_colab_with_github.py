# Step 1: Install required libraries
!pip install streamlit pyngrok -q

# Step 2: Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Step 3: Load and preprocess the dataset
df = pd.read_csv('City_wise_raw_data_1Day_2024_Chennai_1Day.csv')

# Select features and target
features = ['PM10 (µg/m³)', 'NO2 (µg/m³)', 'CO (mg/m³)', 'Ozone (µg/m³)', 'AT (°C)', 'RH (%)', 'WS (m/s)', 'WD (deg)']
X = df[features].fillna(df[features].mean())  # Handle missing values with mean
y = df['PM2.5 (µg/m³)'].fillna(df['PM2.5 (µg/m³)'].mean())

# Step 4: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Train the Random Forest Regressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 8: Save the scaler and model
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(model, 'model.joblib')

# Step 9: Write the Streamlit app code to a file
with open('app.py', 'w') as f:
    f.write('''
import streamlit as st
import pandas as pd
import joblib

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
st.title('PM2.5 Prediction App with Health Provisions')
st.write('Enter the values below to predict PM2.5 levels in Chennai and get health recommendations.')

# Input fields for selected features
pm10 = st.number_input('PM10 (µg/m³)', min_value=0.0, value=50.0)
no2 = st.number_input('NO2 (µg/m³)', min_value=0.0, value=15.0)
co = st.number_input('CO (mg/m³)', min_value=0.0, value=1.0)
ozone = st.number_input('Ozone (µg/m³)', min_value=0.0, value=15.0)
at = st.number_input('Ambient Temperature (°C)', min_value=0.0, value=30.0)
rh = st.number_input('Relative Humidity (%)', min_value=0.0, max_value=100.0, value=75.0)
ws = st.number_input('Wind Speed (m/s)', min_value=0.0, value=2.0)
wd = st.number_input('Wind Direction (deg)', min_value=0.0, max_value=360.0, value=180.0)

# Predict button
if st.button('Predict'):
    # Create input data array
    input_data = [[pm10, no2, co, ozone, at, rh, ws, wd]]
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Get health provisions
    category, provisions = get_health_provisions(prediction)
    
    # Display the prediction
    st.success(f'Predicted PM2.5 Level: {prediction:.2f} µg/m³')
    
    # Display health provisions
    st.subheader(f'Air Quality Category: {category}')
    st.write(f'**Health Provisions**: {provisions}')
    ''')

# Step 10: Create requirements.txt
with open('requirements.txt', 'w') as f:
    f.write('''
streamlit==1.35.0
pandas==2.0.3
numpy==1.25.2
scikit-learn==1.3.0
joblib==1.4.2
    ''')

# Step 11: Create .gitignore
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

# Joblib files (optional, included here for reference)
# scaler.joblib
# model.joblib

# Dataset (optional, comment out if you want to include it)
# City_wise_raw_data_1Day_2024_Chennai_1Day.csv

# Colab-specific
*.ipynb
*.ipynb_checkpoints/

# Misc
.DS_Store
*.log
    ''')

# Step 12: Create README.md
with open('README.md', 'w') as f:
    f.write('''
# PM2.5 Prediction App with Health Provisions

This project is a Streamlit web application that predicts PM2.5 levels in Chennai using a Random Forest Regressor model, based on air quality and weather data. It also provides health recommendations based on the predicted PM2.5 levels.

## Project Structure
- `app.py`: The main Streamlit application code.
- `scaler.joblib`: Saved StandardScaler object for feature scaling.
- `model.joblib`: Saved Random Forest Regressor model.
- `City_wise_raw_data_1Day_2024_Chennai_1Day.csv`: Dataset used for training (optional, not included in repo due to size).
- `requirements.txt`: List of Python dependencies.
- `.gitignore`: Specifies files to exclude from version control.

## Prerequisites
- Python 3.8+
- Git
- Streamlit Cloud or another deployment platform (optional)

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure Model Files**:
   - Place `scaler.joblib` and `model.joblib` in the project root.
   - (Optional) If retraining, include `City_wise_raw_data_1Day_2024_Chennai_1Day.csv` and modify `app.py` to load it.

4. **Run the App Locally**:
   ```bash
   streamlit run app.py
   ```

## Deployment
- **Streamlit Cloud**:
  1. Fork this repository to your GitHub account.
  2. Log in to [Streamlit Cloud](https://streamlit.io/cloud).
  3. Create a new app and link it to this repository.
  4. Ensure `scaler.joblib` and `model.joblib` are included in the repo.
  5. Deploy the app.

## Usage
1. Open the app in a browser.
2. Enter values for PM10, NO2, CO, Ozone, Temperature, Humidity, Wind Speed, and Wind Direction.
3. Click "Predict" to see the predicted PM2.5 level and health recommendations.

## Health Provisions
The app categorizes PM2.5 levels based on WHO/EPA guidelines:
- **Good (0–12 µg/m³)**: Safe for all.
- **Moderate (12.1–35.4 µg/m³)**: Caution for sensitive groups.
- **Unhealthy for Sensitive Groups (35.5–55.4 µg/m³)**: Avoid strenuous outdoor activities.
- **Unhealthy (55.5–150.4 µg/m³)**: Reduce outdoor exposure.
- **Very Unhealthy (150.5–250.4 µg/m³)**: Stay indoors, use N95 masks.
- **Hazardous (>250.4 µg/m³)**: High risk, remain indoors with air purifiers.

## Dataset
The model was trained on `City_wise_raw_data_1Day_2024_Chennai_1Day.csv`, containing daily air quality data for Chennai in 2024. Due to GitHub size limits, it may not be included in the repository.

## License
MIT License

## Author
[Your Name]
    ''')

# Step 13: Ensure dataset is in the directory (already uploaded)
# Note: City_wise_raw_data_1Day_2024_Chennai_1Day.csv should already be in Colab's working directory

print("Project files created: app.py, requirements.txt, .gitignore, README.md, scaler.joblib, model.joblib")
print("Next steps: Follow the GitHub upload instructions below.")