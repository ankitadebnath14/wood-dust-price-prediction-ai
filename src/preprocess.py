import pandas as pd

# Load dataset
df = pd.read_csv("data/wood_dust_data.csv")

# Show first 5 rows
print(df.head())
#check data info
print(df.info())
#check missing value
print(df.isnull().sum())
from sklearn.preprocessing import LabelEncoder

# Create encoder
le = LabelEncoder()

# Encode categorical columns
df['wood_type'] = le.fit_transform(df['wood_type'])
df['season'] = le.fit_transform(df['season'])
df['demand'] = le.fit_transform(df['demand'])

print(df.head())
# Features
X = df.drop("price", axis=1)

# Target
y = df["price"]

print(X.head())
print(y.head())