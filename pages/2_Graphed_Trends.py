import streamlit as st
import gspread
import pandas as pd
import plotly.express as px
from oauth2client.service_account import ServiceAccountCredentials

# 🔹 Page Configuration
st.set_page_config(page_title="Graphed Trends", page_icon="📈")

st.title("📊 Sugar Creek Flow Trends")
st.write("### Live Graph of Predicted CFS from Google Sheets")

# 🔹 Google Sheets Configuration
SHEET_NAME = "sugar_creek_data"
CREDENTIALS_FILE = "gspread_credentials.json"  # Ensure this is in your repo!

# 🔹 Authenticate with Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
client = gspread.authorize(creds)
sheet = client.open(SHEET_NAME).sheet1  # Open the first sheet

# 🔹 Fetch Data
data = sheet.get_all_values()
df = pd.DataFrame(data[1:], columns=data[0])  # Use the first row as column headers

# 🔹 Convert Timestamp Column to Datetime
df["Timestamp (UTC)"] = pd.to_datetime(df["Timestamp (UTC)"])

# 🔹 Convert Prediction Column to Numeric
df["Predicted Sugar Creek CFS"] = pd.to_numeric(df["Predicted Sugar Creek CFS"], errors="coerce")

# 🔹 Sort Data by Time
df = df.sort_values("Timestamp (UTC)")

# 🔹 Display Data
st.write("### 🔍 Latest Data Entries")
st.dataframe(df.tail(10))  # Show the last 10 records

# 🔹 Plot the Data
fig = px.line(df, x="Timestamp (UTC)", y="Predicted Sugar Creek CFS", title="Sugar Creek CFS Predictions Over Time")

st.plotly_chart(fig, use_container_width=True)


