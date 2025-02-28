import os
import pickle
import pandas as pd
import numpy as np
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import xgboost as xgb
import pytz
from datetime import datetime, timezone, timedelta

# 🔹 Google Sheets Configuration
SHEET_NAME = "sugar_creek_data"
CREDENTIALS_FILE = "gspread_credentials.json"  # Ensure this file is in your repo!

# 🔹 Load Google Sheets Credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
client = gspread.authorize(creds)
sheet = client.open(SHEET_NAME).sheet1  # Open the first sheet

# 🔹 Load the trained XGBoost model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "scpm2.pkl")

try:
    with open(MODEL_PATH, "rb") as file:
        xgb_model = pickle.load(file)
except FileNotFoundError:
    print("❌ Model file 'scpm2.pkl' not found.")
    exit(1)

# 🔹 USGS API URLs (Sugar Creek's site ID)
SUGAR_CREEK_SITE = "03588500"  # Update this to the correct Sugar Creek USGS site ID

# 🔹 Function to fetch real-time USGS CFS readings and timestamps
def fetch_real_time_data():
    url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={SUGAR_CREEK_SITE}&parameterCd=00060"
    response = requests.get(url, headers={"Accept": "application/json"})

    if response.status_code == 200:
        try:
            data = response.json()
            time_series = data["value"]["timeSeries"][0]
            latest_value_entry = time_series["values"][0]["value"][0]

            # Extract flow value
            sugar_creek_flow = float(latest_value_entry["value"])

            # Extract & convert timestamp from UTC to Central Time
            raw_timestamp = latest_value_entry["dateTime"]
            parsed_timestamp = datetime.strptime(raw_timestamp, "%Y-%m-%dT%H:%M:%S.%f%z")
            central_tz = pytz.timezone("America/Chicago")
            formatted_timestamp = parsed_timestamp.astimezone(central_tz).strftime("%Y-%m-%d %H:%M:%S")

            return sugar_creek_flow, formatted_timestamp
        except (KeyError, IndexError, TypeError, ValueError):
            return np.nan, "N/A"
    else:
        return np.nan, "N/A"

# 🔹 Fetch Real-Time Data
sugar_creek_cfs, timestamp_str = fetch_real_time_data()

# 🔹 Ensure data is valid before proceeding
if sugar_creek_cfs == np.nan or timestamp_str == "N/A":
    print("❌ Failed to fetch valid USGS data. Exiting.")
    exit(1)

# 🔹 Fetch Historical Data for Lag Values
def fetch_historical_data(reference_timestamp, hours_ago):
    target_timestamp = datetime.strptime(reference_timestamp, "%Y-%m-%d %H:%M:%S") - timedelta(hours=hours_ago)
    start_time = (target_timestamp - timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_time = (target_timestamp + timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ')

    url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={SUGAR_CREEK_SITE}&parameterCd=00060&startDT={start_time}&endDT={end_time}"
    response = requests.get(url, headers={"Accept": "application/json"})

    if response.status_code == 200:
        try:
            data = response.json()
            time_series = data["value"]["timeSeries"][0]
            values = time_series["values"][0]["value"]

            closest_value, closest_time = np.nan, "N/A"
            min_time_diff = float("inf")

            for entry in values:
                entry_time = datetime.strptime(entry["dateTime"], "%Y-%m-%dT%H:%M:%S.%f%z").astimezone(pytz.utc)
                time_diff = abs((entry_time - target_timestamp.astimezone(pytz.utc)).total_seconds())

                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_value = float(entry["value"])
                    closest_time = entry_time.astimezone(pytz.timezone("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")

            return closest_value
        except (KeyError, IndexError, TypeError, ValueError):
            return np.nan
    else:
        return np.nan

# 🔹 Fetch historical lag values
lag_1 = fetch_historical_data(timestamp_str, 24)
lag_3 = fetch_historical_data(timestamp_str, 72)
lag_7 = fetch_historical_data(timestamp_str, 168)

# 🔹 Prepare Model Input (Ensure all required features are present)
model_input = pd.DataFrame([{
    "Shoal_Creek": sugar_creek_cfs,
    "Shoal_Creek_Lag1": lag_1,
    "Shoal_Creek_Lag3": lag_3,
    "Shoal_Creek_Lag7": lag_7
}])

# Ensure model input matches trained model's expected features
for feature in xgb_model.feature_names_in_:
    if feature not in model_input.columns:
        model_input[feature] = np.nan  # Add missing columns with NaN values

# Select only the required columns
model_input = model_input[xgb_model.feature_names_in_]

# 🔹 Run Prediction
prediction = xgb_model.predict(model_input)[0]

# 🔹 Check for duplicates in Google Sheets before appending
existing_data = sheet.get_all_values()
timestamps_in_sheet = [row[0] for row in existing_data[1:]]  # Skip header row

if timestamp_str not in timestamps_in_sheet:
    # ✅ Append new row with only Timestamp and Predicted Sugar Creek CFS
    sheet.append_row([timestamp_str, float(prediction)])
    print(f"✅ Recorded: {timestamp_str} - Sugar Creek Prediction: {prediction:.2f} CFS")
else:
    print(f"⚠️ Duplicate entry detected. Skipping {timestamp_str}")

