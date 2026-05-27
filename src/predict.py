import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = joblib.load("models/model.pkl")

# Load dataset
df = pd.read_csv("data/wood_dust_data.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Clean text values
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Create encoders
wood_encoder = LabelEncoder()
season_encoder = LabelEncoder()
demand_encoder = LabelEncoder()

# Fit encoders
wood_encoder.fit(df['wood_type'])
season_encoder.fit(df['season'])
demand_encoder.fit(df['demand'])

# ===== USER INPUT =====

wood_type = "Teak"
moisture = 10
quantity = 600
transport_cost = 2200
season = "Winter"
demand = "High"

# Encode categorical inputs
wood_type_encoded = wood_encoder.transform([wood_type])[0]
season_encoded = season_encoder.transform([season])[0]
demand_encoded = demand_encoder.transform([demand])[0]

# Create input dataframe
input_data = pd.DataFrame({
    'wood_type': [wood_type_encoded],
    'moisture': [moisture],
    'quantity': [quantity],
    'transport_cost': [transport_cost],
    'season': [season_encoded],
    'demand': [demand_encoded]
})

# Predict
prediction = model.predict(input_data)

# Show result
print("Predicted Price:", prediction[0])