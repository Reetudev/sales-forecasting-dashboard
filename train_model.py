import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv("dataset/train.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

# Group sales by date
daily_sales = df.groupby("date")["sales"].sum().reset_index()

# Create time index
daily_sales["day_number"] = range(len(daily_sales))

# Features and Target
X = daily_sales[["day_number"]]
y = daily_sales["sales"]

# Train Model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "forecasting_model.pkl")

# Save processed data
daily_sales.to_csv("processed_sales.csv", index=False)

print("✅ Forecasting Model Saved Successfully")