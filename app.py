import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


st.set_page_config(
    page_title="Wood Dust Price Predictor",
    page_icon="📈",
    layout="wide"
)

# =========================
# BASE DIRECTORY
# =========================

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# LOAD MODEL
# =========================

model_path = os.path.join(BASE_DIR, "models", "model.pkl")

model = joblib.load(model_path)

# =========================
# LOAD DATASET
# =========================

data_path = os.path.join(BASE_DIR, "data", "wood_dust_data.csv")

df = pd.read_csv(data_path)

# Clean column names
df.columns = df.columns.str.strip()

# Clean text values
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# =========================
# ENCODERS
# =========================

wood_encoder = LabelEncoder()
season_encoder = LabelEncoder()
demand_encoder = LabelEncoder()

wood_encoder.fit(df['wood_type'])
season_encoder.fit(df['season'])
demand_encoder.fit(df['demand'])

# =========================
# APP TITLE
# =========================

st.title("AI-Based Wood Dust Price Prediction System")

st.write("Enter wood dust details to predict price.")
st.sidebar.header("Project Information")

st.sidebar.write("""
This AI system predicts wood dust prices based on:

- Wood Type
- Moisture
- Quantity
- Transport Cost
- Season
- Market Demand
""")
# =========================
# USER INPUTS
# =========================

col1, col2 = st.columns(2)

with col1:
    wood_type = st.selectbox(
        "Select Wood Type",
        df['wood_type'].unique()
    )

    moisture = st.number_input(
        "Moisture (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0
    )

    quantity = st.number_input(
        "Quantity (kg)",
        min_value=1,
        value=500
    )

with col2:
    transport_cost = st.number_input(
        "Transport Cost",
        min_value=0.0,
        value=2000.0
    )

    season = st.selectbox(
        "Select Season",
        df['season'].unique()
    )

    demand = st.selectbox(
        "Select Demand Level",
        df['demand'].unique()
    )

# =========================
# PREDICT BUTTON
# =========================

if st.button("Predict Price"):

    # Encode inputs
    wood_encoded = wood_encoder.transform([wood_type])[0]
    season_encoded = season_encoder.transform([season])[0]
    demand_encoded = demand_encoder.transform([demand])[0]

    # Create dataframe
    input_data = pd.DataFrame({
        'wood_type': [wood_encoded],
        'moisture': [moisture],
        'quantity': [quantity],
        'transport_cost': [transport_cost],
        'season': [season_encoded],
        'demand': [demand_encoded]
    })

    # Prediction
    prediction = model.predict(input_data)
    st.metric(
    label="Predicted Wood Dust Price",
    value=f"₹{prediction[0]:,.2f}"
    )
    # Display result
    st.metric(
    label="Predicted Wood Dust Price",
    value=f"₹{prediction[0]:,.2f}"
    )

#business insight logic
    if prediction[0] > 8000:
        st.success("High market value wood dust detected.")

    elif prediction[0] > 5000:
        st.warning("Moderate market value.")

    else:
        st.error("Low market value wood dust.")
#data visualisation
st.subheader("Dataset Preview")

st.dataframe(df.head())

#chart
st.subheader("Price Distribution")

st.bar_chart(df['price'])
# CUSTOMER HEADER
st.markdown("""
# 🌲 Smart Wood Dust AI System
### AI-Powered Price Prediction for Wood Dust Businesses
""")
st.markdown("---")
st.write("Developed using Machine Learning and Streamlit")