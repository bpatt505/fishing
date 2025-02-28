import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="Sugar Creek Data Lookup", page_icon="🎣")

st.title("Welcome to the Fishing Data Lookup App!")
st.write(
    "Use the sidebar to navigate between real-time lookup and historical data analysis."
)


# ✅ Load the trained XGBoost model (Ensure correct path)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "scpm2.pkl")

# Load the model safely
try:
    with open(MODEL_PATH, "rb") as file:
        xgb_model = pickle.load(file)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model file 'scpm2.pkl' not found. Ensure it's in the root directory.")

# USGS API URLs for real-time data
USGS_SITES = {
    "Shoal_Creek": "03588500",
    "Big_Nance_Creek": "03586500",
    "Limestone_Creek": "03576250",
    "Swan_Creek": "03577225",
}

# Function to fetch real-time USGS CFS readings and timestamps
def fetch_real_time_data():
    real_time_values = {}
    timestamps = {}

    for creek, site in USGS_SITES.items():
        url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={site}&parameterCd=00060"
        response = requests.get(url, headers={"Accept": "application/json"})
        
        if response.status_code == 200:
            try:
                data = response.json()
                time_series = data["value"]["timeSeries"][0]
                latest_value_entry = time_series["values"][0]["value"][0]
                
                real_time_values[creek] = float(latest_value_entry["value"])
                raw_timestamp = latest_value_entry["dateTime"]
                
                # Convert timezone-aware timestamp to UTC
                parsed_timestamp = datetime.strptime(raw_timestamp, "%Y-%m-%dT%H:%M:%S.%f%z").astimezone(timezone.utc)
                formatted_timestamp = parsed_timestamp.astimezone().strftime("%m/%d/%Y %I:%M %p")
                timestamps[creek] = formatted_timestamp
            except (KeyError, IndexError, TypeError, ValueError) as e:
                real_time_values[creek] = np.nan
                timestamps[creek] = "N/A"
        else:
            real_time_values[creek] = np.nan
            timestamps[creek] = "N/A"

    return real_time_values, timestamps

# Function to fetch historical lag values using the real-time timestamp as reference
def fetch_historical_data(site, reference_timestamp, hours_ago):
    if isinstance(reference_timestamp, str):
        reference_timestamp = datetime.strptime(reference_timestamp, "%m/%d/%Y %I:%M %p").astimezone(timezone.utc)
    target_timestamp = reference_timestamp - timedelta(hours=hours_ago)
    start_time = (target_timestamp - timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_time = (target_timestamp + timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={site}&parameterCd=00060&startDT={start_time}&endDT={end_time}"
    response = requests.get(url, headers={"Accept": "application/json"})
    
    if response.status_code == 200:
        try:
            data = response.json()
            time_series = data["value"]["timeSeries"][0]
            values = time_series["values"][0]["value"]
            
            closest_value, closest_time = np.nan, "N/A"
            min_time_diff = float("inf")
            
            for entry in values:
                entry_time = datetime.strptime(entry["dateTime"], "%Y-%m-%dT%H:%M:%S.%f%z").astimezone(timezone.utc)
                time_diff = abs((entry_time - target_timestamp).total_seconds())
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_value = float(entry["value"])
                    closest_time = entry_time.astimezone().strftime("%m/%d/%Y %I:%M %p")
            
            return closest_value, closest_time
        except (KeyError, IndexError, TypeError, ValueError) as e:
            return np.nan, "N/A"
    else:
        return np.nan, "N/A"

st.title("Sugar Creek CFS Prediction")
st.write("### Predicting Sugar Creek's Flow Using Real-Time USGS Data")

if st.button("Get Prediction"):
    st.write("Fetching Real-Time and Historical Data from USGS...")
    
    real_time_data, timestamps = fetch_real_time_data()
    
    lag_data = {}
    lag_timestamps = {}
    
    for creek, site in USGS_SITES.items():
        if timestamps[creek] != "N/A":
            reference_timestamp = datetime.strptime(timestamps[creek], "%m/%d/%Y %I:%M %p").astimezone(timezone.utc)
            lag_data[f"{creek}_Lag1"], lag_timestamps[f"{creek}_Lag1"] = fetch_historical_data(site, reference_timestamp, 24)
            lag_data[f"{creek}_Lag3"], lag_timestamps[f"{creek}_Lag3"] = fetch_historical_data(site, reference_timestamp, 72)
            lag_data[f"{creek}_Lag7"], lag_timestamps[f"{creek}_Lag7"] = fetch_historical_data(site, reference_timestamp, 168)
    
    with st.expander("🌊 USGS CFS Readings at Selected Time"):
        for creek, value in real_time_data.items():
            st.write(f"**{creek.replace('_', ' ')}**: {value} CFS *(Recorded at: {timestamps[creek]})*")
    
    with st.expander("🕰 Historical Lag Data"):
        for key, value in lag_data.items():
            st.write(f"**{key.replace('_', ' ')}**: {value} CFS *(Recorded at: {lag_timestamps[key]})*")
    
    model_input = pd.DataFrame([{**real_time_data, **lag_data}])
    model_input = model_input[[col for col in xgb_model.feature_names_in_]]
    prediction = xgb_model.predict(model_input)[0]
    
    st.write("### 📊 **Predicted Sugar Creek CFS**")
    st.success(f"Predicted Flow: {prediction:.2f} CFS")
