import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("data/wood_dust_data.csv")

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Remove spaces from text values
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Encode categorical columns
le = LabelEncoder()

df['wood_type'] = le.fit_transform(df['wood_type'])
df['season'] = le.fit_transform(df['season'])
df['demand'] = le.fit_transform(df['demand'])

# Features and target
X = df.drop("price", axis=1)
y = df["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestRegressor()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Save model
joblib.dump(model, "models/model.pkl")

print("Model saved successfully!")