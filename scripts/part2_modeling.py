import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aavail.data_ingestion import load_aavail_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

# ---------- 1. Load & prepare ----------
df = load_aavail_data("cs-train")
df["year"] = df["year"].astype(int)
df["month"] = df["month"].astype(int)
df["day"] = pd.to_numeric(df.get("day", 1), errors="coerce").fillna(1).astype(int)
df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))

monthly_rev = df.groupby(df["date"].dt.to_period("M"))["price"].sum().to_timestamp().sort_index()

# Train/test split (last 20% as test)
train_size = int(len(monthly_rev) * 0.8)
train = monthly_rev.iloc[:train_size]
test = monthly_rev.iloc[train_size:]

results = []

# ---------- 2. ARIMA ----------
arima_order = (3,1,1)
arima = sm.tsa.ARIMA(train, order=arima_order).fit()
pred_arima = arima.forecast(steps=len(test))
results.append(["ARIMA(3,1,1)",
                mean_squared_error(test, pred_arima, squared=False),
                mean_absolute_error(test, pred_arima)])

# ---------- 3. SARIMAX ----------
sarimax = sm.tsa.statespace.SARIMAX(train,
                                    order=(1,1,1),
                                    seasonal_order=(1,1,1,12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False).fit()
pred_sarimax = sarimax.forecast(steps=len(test))
results.append(["SARIMAX(1,1,1)(1,1,1,12)",
                mean_squared_error(test, pred_sarimax, squared=False),
                mean_absolute_error(test, pred_sarimax)])

# ---------- 4. Holt-Winters Exponential Smoothing ----------
holt = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
pred_holt = holt.forecast(len(test))
results.append(["Holt-Winters",
                mean_squared_error(test, pred_holt, squared=False),
                mean_absolute_error(test, pred_holt)])

# ---------- 5. Prophet ----------
prophet_df = pd.DataFrame({"ds": monthly_rev.index, "y": monthly_rev.values})
train_prophet = prophet_df.iloc[:train_size]
test_prophet = prophet_df.iloc[train_size:]

m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
m.fit(train_prophet)
future = m.make_future_dataframe(periods=len(test), freq='MS')
forecast = m.predict(future)
pred_prophet = forecast.tail(len(test))["yhat"].values
results.append(["Prophet",
                mean_squared_error(test, pred_prophet, squared=False),
                mean_absolute_error(test, pred_prophet)])

# ---------- 6. Show table ----------
results_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE"])
print("\n=== Time Series Model Comparison ===")
print(results_df)

# ---------- 7. Plot best model ----------
best_model = results_df.sort_values("RMSE").iloc[0]["Model"]
print(f"\nBest model: {best_model}")

plt.figure(figsize=(10,5))
plt.plot(monthly_rev.index, monthly_rev.values, label="Actual")

if best_model.startswith("ARIMA"): plt.plot(test.index, pred_arima, "--", label=best_model)
if best_model.startswith("SARIMAX"): plt.plot(test.index, pred_sarimax, "--", label=best_model)
if best_model.startswith("Holt"): plt.plot(test.index, pred_holt, "--", label=best_model)
if best_model == "Prophet": plt.plot(test.index, pred_prophet, "--", label=best_model)

plt.legend()
plt.title(f"Best Time-Series Model: {best_model}")
plt.show()

# --- Save the chosen model ---
if best_model.startswith("Holt"):
    final_model = holt
elif best_model.startswith("SARIMAX"):
    final_model = sarimax
elif best_model.startswith("ARIMA"):
    final_model = arima
elif best_model == "Prophet":
    final_model = m
else:
    final_model = holt  # fallback

joblib.dump(final_model, "holtwinters_model.pkl")
print("Model saved to holtwinters_model.pkl")
