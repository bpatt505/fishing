import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta, timezone

# ✅ Set page title and icon for Streamlit multipage app
st.set_page_config(page_title="Observed Trends", page_icon="📈")

st.title("Observed CFS Data Prediction Trend (coming soon)")

