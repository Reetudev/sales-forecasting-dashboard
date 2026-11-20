import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


df = pd.read_csv("dataset/train.csv")

df["date"] = pd.to_datetime(df["date"], dayfirst=True)

daily_sales = df.groupby("date")["sales"].sum().reset_index()


daily_sales["day_number"] = range(len(daily_sales))


X = daily_sales[["day_number"]]
y = daily_sales["sales"]


model = LinearRegression()
model.fit(X, y)


joblib.dump(model, "forecasting_model.pkl")


daily_sales.to_csv("processed_sales.csv", index=False)

print("✅ Forecasting Model Saved Successfully")
