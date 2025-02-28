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
CREDENTIALS_FILE = "gspread_credentials.json"

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

# 🔹 USGS Creek Sites (same ones used in the model)
USGS_SITES = {
    "Shoal_Creek": "03588500",
    "Big_Nance_Creek": "03586500",
    "Limestone_Creek": "03576250",
    "Swan_Creek": "03577225",
}

# 🔹 Function to fetch real-time USGS CFS readings and timestamps
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

                # Convert to UTC and then to Central Time
                parsed_timestamp = datetime.strptime(raw_timestamp, "%Y-%m-%dT%H:%M:%S.%f%z")
                central_tz = pytz.timezone("America/Chicago")
                formatted_timestamp = parsed_timestamp.astimezone(central_tz).strftime("%Y-%m-%d %H:%M:%S")

                timestamps[creek] = formatted_timestamp
            except (KeyError, IndexError, TypeError, ValueError):
                real_time_values[creek] = np.nan
                timestamps[creek] = "N/A"
        else:
            real_time_values[creek] = np.nan
            timestamps[creek] = "N/A"

    return real_time_values, timestamps

# 🔹 Fetch Historical Data
def fetch_historical_data(site, reference_timestamp, hours_ago):
    target_timestamp = datetime.strptime(reference_timestamp, "%Y-%m-%d %H:%M:%S") - timedelta(hours=hours_ago)
    start_time = (target_timestamp - timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_time = (target_timestamp + timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ')

    url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={site}&parameterCd=00060&startDT={start_time}&endDT={end_time}"
    response = requests.get(url, headers={"Accept": "application/json"})

    if response.status_code == 200:
        try:
            data = response.json()
            time_series = data["value"]["timeSeries"][0]
            values = time_series["values"][0]["value"]

            closest_value, min_time_diff = np.nan, float("inf")

            for entry in values:
                entry_time = datetime.strptime(entry["dateTime"], "%Y-%m-%dT%H:%M:%S.%f%z").astimezone(pytz.utc)
                time_diff = abs((entry_time - target_timestamp.astimezone(pytz.utc)).total_seconds())

                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_value = float(entry["value"])

            return closest_value
        except (KeyError, IndexError, TypeError, ValueError):
            return np.nan
    else:
        return np.nan

# 🔹 Fetch Real-Time Data
real_time_data, timestamps = fetch_real_time_data()

# 🔹 Ensure we have a valid timestamp (Shoal Creek used as reference)
reference_timestamp = timestamps.get("Shoal_Creek", "N/A")
if reference_timestamp == "N/A":
    print("❌ No valid timestamp found. Exiting.")
    exit(1)

# 🔹 Fetch historical lag values for **all relevant creeks**
lag_data = {}
for creek, site in USGS_SITES.items():
    if reference_timestamp != "N/A":
        lag_data[f"{creek}_Lag1"] = fetch_historical_data(site, reference_timestamp, 24)
        lag_data[f"{creek}_Lag3"] = fetch_historical_data(site, reference_timestamp, 72)
        lag_data[f"{creek}_Lag7"] = fetch_historical_data(site, reference_timestamp, 168)

# 🔹 Prepare Model Input
model_input = pd.DataFrame([{**real_time_data, **lag_data}])

# Ensure model input matches trained model's expected features
for feature in xgb_model.feature_names_in_:
    if feature not in model_input.columns:
        model_input[feature] = np.nan  # Add missing columns with NaN values

# Select only the required columns
model_input = model_input[xgb_model.feature_names_in_]

# 🔹 Run Prediction
prediction = xgb_model.predict(model_input)[0]

# 🔹 Store only Sugar Creek data in Google Sheets
existing_data = sheet.get_all_values()
timestamps_in_sheet = [row[0] for row in existing_data[1:]]  # Skip header row

if reference_timestamp not in timestamps_in_sheet:
    # ✅ Append only one row per prediction
    sheet.append_row([reference_timestamp, float(prediction)])
    print(f"✅ Recorded: {reference_timestamp} - Sugar Creek Prediction: {prediction:.2f} CFS")
else:
    print(f"⚠️ Duplicate entry detected. Skipping {reference_timestamp}")
