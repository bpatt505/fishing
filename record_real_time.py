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

# 🔹 Load Google Sheets Credentials from Environment
google_creds = os.getenv("GOOGLE_SHEETS_CREDENTIALS")
if google_creds:
    with open(CREDENTIALS_FILE, "w") as f:
        f.write(google_creds)

# 🔹 Authenticate with Google Sheets
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

# 🔹 USGS API URLs for real-time data (ONLY needed creeks for prediction)
USGS_SITES = {
    "Shoal_Creek": "03588500",
    "Big_Nance_Creek": "03586500",
    "Limestone_Creek": "03576250",
    "Swan_Creek": "03577225",
}

# 🔹 Function to fetch real-time USGS CFS readings
def fetch_real_time_data():
    real_time_values = {}

    for creek, site in USGS_SITES.items():
        url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={site}&parameterCd=00060"
        response = requests.get(url, headers={"Accept": "application/json"})
        
        if response.status_code == 200:
            try:
                data = response.json()
                time_series = data["value"]["timeSeries"][0]
                latest_value_entry = time_series["values"][0]["value"][0]
                real_time_values[creek] = float(latest_value_entry["value"])
            except (KeyError, IndexError, TypeError, ValueError):
                real_time_values[creek] = np.nan
        else:
            real_time_values[creek] = np.nan

    return real_time_values

# 🔹 Fetch Real-Time Data
real_time_data = fetch_real_time_data()

# 🔹 Prepare Model Input (ONLY the required features)
model_input = pd.DataFrame([real_time_data])

# 🔹 Ensure all required columns exist before model prediction
for feature in xgb_model.feature_names_in_:
    if feature not in model_input.columns:
        model_input[feature] = np.nan  # Add missing columns with NaN values

# Select only the expected columns
model_input = model_input[xgb_model.feature_names_in_]

# 🔹 Run Prediction
prediction = xgb_model.predict(model_input)[0]

# 🔹 Get Central Time for timestamp
central_tz = pytz.timezone("America/Chicago")
timestamp_str = datetime.now(timezone.utc).astimezone(central_tz).strftime("%Y-%m-%d %H:%M:%S")

# 🔹 Fetch existing timestamps to avoid duplicates
existing_data = sheet.get_all_values()
timestamps_in_sheet = [row[0] for row in existing_data[1:]]  # Skip header row

if timestamp_str not in timestamps_in_sheet:
    # ✅ Append new row with only Timestamp and Predicted Sugar Creek CFS
    sheet.append_row([timestamp_str, float(prediction)])

print("✅ Sugar Creek prediction successfully recorded to Google Sheets.")
