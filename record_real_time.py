import json
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

# ✅ Load Google Sheets Credentials from ENV variable
try:
    credentials_json = os.getenv("GOOGLE_SHEETS_CREDENTIALS")
    if not credentials_json:
        raise ValueError("GOOGLE_SHEETS_CREDENTIALS environment variable is missing!")

    creds_dict = json.loads(credentials_json)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"])

    client = gspread.authorize(creds)
    sheet = client.open(SHEET_NAME).sheet1  # Open first sheet

except Exception as e:
    print(f"❌ Error loading Google Sheets credentials: {e}")
    exit(1)

# 🔹 Load the trained XGBoost model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "scpm2.pkl")

try:
    with open(MODEL_PATH, "rb") as file:
        xgb_model = pickle.load(file)
except FileNotFoundError:
    print("❌ Model file 'scpm2.pkl' not found.")
    exit(1)

# 🔹 USGS API URLs for real-time data
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
                
                # Convert timezone-aware timestamp to UTC
                parsed_timestamp = datetime.strptime(raw_timestamp, "%Y-%m-%dT%H:%M:%S.%f%z").astimezone(timezone.utc)
                formatted_timestamp = parsed_timestamp.strftime("%m/%d/%Y %I:%M %p")
                timestamps[creek] = formatted_timestamp
            except (KeyError, IndexError, TypeError, ValueError):
                real_time_values[creek] = np.nan
                timestamps[creek] = "N/A"
        else:
            real_time_values[creek] = np.nan
            timestamps[creek] = "N/A"

    return real_time_values, timestamps

# 🔹 Fetch Real-Time Data
real_time_data, timestamps = fetch_real_time_data()

# 🔹 Prepare Model Input
model_input = pd.DataFrame([real_time_data])
model_input = model_input[[col for col in xgb_model.feature_names_in_]]

# 🔹 Run Prediction
prediction = xgb_model.predict(model_input)[0]

# 🔹 Save Data to Google Sheets
# Define the US Central Time Zone
central_tz = pytz.timezone('America/Chicago')

# Convert UTC to Central Time
timestamp = datetime.now(timezone.utc).astimezone(central_tz).strftime("%Y-%m-%d %H:%M:%S")

# ✅ Fetch all existing timestamps to check for duplicates
existing_data = sheet.get_all_values()
timestamps = [row[0] for row in existing_data[1:]]  # Skip header row

if timestamp in timestamps:
    # ✅ If timestamp exists, update the existing row
    row_index = timestamps.index(timestamp) + 2  # Offset for 1-based index & header
    sheet.update(f"A{row_index}:C{row_index}", [[timestamp, "Sugar_Creek_Prediction", float(prediction)]])
else:
    # ✅ If timestamp does not exist, append a new row
    sheet.append_row([timestamp, "Sugar_Creek_Prediction", float(prediction)])

print("✅ Data successfully recorded to Google Sheets.")
