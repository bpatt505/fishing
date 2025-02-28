import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta, timezone

# ✅ Set page title and icon for Streamlit multipage app
st.set_page_config(page_title="Historical Lookup", page_icon="⏰")

st.title("Sugar Creek Historical Lookup")

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

# Function to fetch historical CFS data from USGS

def fetch_usgs_data(site, target_timestamp):
    url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={site}&parameterCd=00060&startDT={target_timestamp}&endDT={target_timestamp}"
    response = requests.get(url, headers={"Accept": "application/json"})
    
    if response.status_code == 200:
        try:
            data = response.json()
            time_series = data["value"]["timeSeries"][0]
            values = time_series["values"][0]["value"]
            
            closest_value = float(values[-1]["value"]) if values else np.nan
            return closest_value
        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f"⚠️ Error fetching USGS data for site {site}: {e}")
            return np.nan
    else:
        print(f"❌ Failed to fetch USGS data for site {site}. HTTP Status: {response.status_code}")
        return np.nan

# Streamlit UI - User Inputs
st.write("### Enter a Past Date and Time to Predict Sugar Creek's Flow")

# Get current UTC date (limit user selection to past & present dates)
today = datetime.utcnow().date()

# Date input with pop-up calendar (MM/DD/YYYY format for user convenience)
selected_date = st.date_input("Select Date", today - timedelta(days=1), min_value=datetime(2000, 1, 1).date(), max_value=today)
selected_date_str = selected_date.strftime('%m/%d/%Y')

# Time input with AM/PM selection
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    selected_hour = st.selectbox("Hour", [str(i) for i in range(1, 13)], index=11)
with col2:
    selected_minute = st.selectbox("Minute", ["00", "15", "30", "45"], index=0)
with col3:
    am_pm = st.selectbox("AM/PM", ["AM", "PM"], index=1)

selected_time = f"{selected_hour}:{selected_minute}"

# Convert selected time to 24-hour format
hour_24 = int(selected_hour)
if am_pm == "PM" and hour_24 != 12:
    hour_24 += 12
elif am_pm == "AM" and hour_24 == 12:
    hour_24 = 0

# Combine date and time into datetime object
selected_datetime = datetime.combine(selected_date, datetime.min.time()).replace(hour=hour_24, minute=int(selected_minute))

# Convert to UTC (assuming input is local time)
selected_datetime_utc = selected_datetime.astimezone(timezone.utc)

# Format for USGS API
usgs_formatted_datetime = selected_datetime_utc.strftime('%Y-%m-%dT%H:%M:%SZ')  # Explicit UTC conversion before formatting

# Compute Lag1, Lag3, and Lag7 timestamps
lag1_datetime_utc = (selected_datetime_utc - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
lag3_datetime_utc = (selected_datetime_utc - timedelta(days=3)).strftime('%Y-%m-%dT%H:%M:%SZ')
lag7_datetime_utc = (selected_datetime_utc - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%SZ')

with st.expander("📊 Debugging & Logged Data"):
    st.write("#### Selected Inputs & Converted Timestamps")
    st.write(f"Selected Date: {selected_date_str}")
    st.write(f"Selected Time: {selected_hour}:{selected_minute} {am_pm}")
    st.write(f"Converted 24-Hour Time: {hour_24}:{selected_minute}")
    st.write(f"Selected Datetime (Local): {selected_datetime}")
    st.write(f"Selected Datetime (UTC for USGS API): {selected_datetime_utc}")
    st.write(f"USGS Formatted Datetime: {usgs_formatted_datetime}")
    st.write(f"Lag1 Datetime (UTC for USGS API): {lag1_datetime_utc}")
    st.write(f"Lag3 Datetime (UTC for USGS API): {lag3_datetime_utc}")
    st.write(f"Lag7 Datetime (UTC for USGS API): {lag7_datetime_utc}")

# Button to fetch USGS data
if st.button("Fetch Historic Data"):
    st.write("Fetching historical data for the selected timestamp...")
    
    # Fetch USGS data for selected and lag timestamps
    real_time_data = {}
    for creek, site in USGS_SITES.items():
        real_time_data[creek] = fetch_usgs_data(site, usgs_formatted_datetime)
        real_time_data[f"{creek}_Lag1"] = fetch_usgs_data(site, lag1_datetime_utc)
        real_time_data[f"{creek}_Lag3"] = fetch_usgs_data(site, lag3_datetime_utc)
        real_time_data[f"{creek}_Lag7"] = fetch_usgs_data(site, lag7_datetime_utc)

    # Prepare data for prediction
    model_input = pd.DataFrame([{**real_time_data}])

    # Ensure column order matches the trained model
    model_input = model_input[[col for col in xgb_model.feature_names_in_]]

    # Run prediction
    prediction = xgb_model.predict(model_input)[0]
    
    # Display fetched values
    with st.expander("🌊 USGS CFS Readings at Selected Time"):
        for creek, value in real_time_data.items():
            # Determine appropriate timestamp for each entry
            timestamp = selected_datetime.strftime('%m/%d/%Y at %I:%M %p')
            if 'Lag1' in creek:
                timestamp = (selected_datetime - timedelta(days=1)).strftime('%m/%d/%Y at %I:%M %p')
            elif 'Lag3' in creek:
                timestamp = (selected_datetime - timedelta(days=3)).strftime('%m/%d/%Y at %I:%M %p')
            elif 'Lag7' in creek:
                timestamp = (selected_datetime - timedelta(days=7)).strftime('%m/%d/%Y at %I:%M %p')
            st.write(f"**{creek.replace('_', ' ')}**: {value} CFS (Recorded at: {timestamp})")
    
    # Display prediction result
    st.success(f"Predicted Flow: {prediction:.2f} CFS on {selected_date_str} at {selected_hour}:{selected_minute} {am_pm}")
    
