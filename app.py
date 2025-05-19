import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime
import plotly.graph_objects as go
from io import StringIO

# --- Constants ---
MODEL_FILE = 'energy_demand_model.pk1'
LOG_FILE = 'model_log.txt'

# --- Model Training Function ---
@st.cache_data
def load_data():
    data = pd.read_csv('weather_data_all1.csv')
    data['Time'] = pd.to_datetime(data['Time'])
    data['Hour'] = data['Time'].dt.hour
    data['DayOfWeek'] = data['Time'].dt.dayofweek
    data['Month'] = data['Time'].dt.month
    data.dropna(inplace=True)
    return data

@st.cache_resource
def train_models():
    data = load_data()
    X = data[['Hour', 'DayOfWeek', 'Month', 'Temp (F)', 'Humidity']]
    y = data['Wind Speed']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        models[name] = {
            'model': model,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
    
    return models, X_test, y_test

# --- Temperature Conversion ---
def f_to_c(f_temp):
    return (f_temp - 32) * 5/9

def c_to_f(c_temp):
    return (c_temp * 9/5) + 32

# --- Streamlit App ---
st.set_page_config(page_title="Energy Demand Predictor", layout="wide")
st.title("🌍 Nigeria Energy Demand Predictor")
st.write("Predict wind speed using weather data and compare ML models")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Input Parameters")
    temp_unit = st.radio("Temperature Unit", ("°F", "°C"), index=0)
    
    if temp_unit == "°F":
        temp_value = st.slider("Temperature", 50.0, 110.0, 77.0)
    else:
        temp_value = st.slider("Temperature", 10.0, 43.0, 25.0)
        temp_value = c_to_f(temp_value)
    
    hour = st.slider("Hour", 0, 23, 12)
    day_of_week = st.selectbox("Day of Week", 
                             ["Monday", "Tuesday", "Wednesday", 
                              "Thursday", "Friday", "Saturday", "Sunday"],
                             index=0)
    month = st.selectbox("Month", 
                        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                        index=5)
    humidity = st.slider("Humidity (%)", 0, 100, 50)

# --- Model Loading/Training ---
models, X_test, y_test = train_models()
selected_model = st.selectbox("Choose Model", list(models.keys()))

# --- Prediction ---
if st.button("🚀 Predict Wind Speed"):
    # Prepare input
    day_map = {"Monday":0, "Tuesday":1, "Wednesday":2, 
               "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
    month_map = {month:i+1 for i, month in enumerate(
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
    
    input_data = np.array([[
        hour,
        day_map[day_of_week],
        month_map[month],
        temp_value,
        humidity
    ]])
    
    # Get prediction
    model_info = models[selected_model]
    prediction = model_info['model'].predict(input_data)[0]
    
    # Display results
    st.success(f"Predicted Wind Speed: **{prediction:.2f} mph**")
    st.metric("Model RMSE", f"{model_info['rmse']:.2f}")
    st.metric("R² Score", f"{model_info['r2']:.2f}")
    
    # --- Interactive Plot ---
    y_pred = model_info['model'].predict(X_test)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='royalblue', opacity=0.6)
    ))
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title=f"{selected_model} Performance",
        xaxis_title="Actual Wind Speed (mph)",
        yaxis_title="Predicted Wind Speed (mph)",
        hovermode='closest'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Downloadable Report ---
    report = f"""
    Energy Demand Prediction Report
    -------------------------------
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Model: {selected_model}
    Input Parameters:
    - Temperature: {temp_value:.1f}°F ({f_to_c(temp_value):.1f}°C)
    - Hour: {hour}
    - Day: {day_of_week}
    - Month: {month}
    - Humidity: {humidity}%
    
    Results:
    - Predicted Wind Speed: {prediction:.2f} mph
    - Model RMSE: {model_info['rmse']:.2f}
    - R² Score: {model_info['r2']:.2f}
    """
    
    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="wind_speed_prediction_report.txt",
        mime="text/plain"
    )
    
    # Log prediction
    log_entry = f"{datetime.now()},{selected_model},{temp_value},{hour},{day_of_week},{month},{humidity},{prediction}\n"
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)

# --- Model Comparison Section ---
st.header("📊 Model Comparison")
model_names = list(models.keys())
rmse_values = [models[name]['rmse'] for name in model_names]
r2_values = [models[name]['r2'] for name in model_names]

col1, col2 = st.columns(2)
with col1:
    st.metric("Best RMSE", f"{min(rmse_values):.2f}", 
             delta=f"{model_names[np.argmin(rmse_values)]}")
with col2:
    st.metric("Best R²", f"{max(r2_values):.2f}", 
             delta=f"{model_names[np.argmax(r2_values)]}")

# --- Data Exploration ---
if st.checkbox("Show raw data"):
    st.dataframe(load_data(), height=300)