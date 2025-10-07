# IBM AI Enterprise Workflow Capstone — Part 1

## 1. Business Scenario

AAVAIL is a subscription-based video streaming company operating across several international markets. Management would like to **forecast next month’s revenue** in order to improve financial planning, optimize marketing spend, and allocate resources more effectively. Historical transaction records (new subscriptions, renewals, cancellations) are available and can be leveraged to build predictive models that anticipate revenue trends.

## 2. Testable Hypotheses

1. **H1 — Seasonality:** Monthly revenue exhibits seasonal patterns (e.g., spikes during holidays such as December or January).
2. **H2 — Autoregression:** Revenue in the upcoming month can be reliably predicted using previous months’ revenue (e.g., 1-month, 3-month, or 6-month lags).
3. **H3 — Customer dynamics:** Changes in the number of active, new, and churned customers have a direct and measurable impact on revenue.
4. **H4 — Promotions and anomalies:** Promotional campaigns or discounts cause revenue surges that appear as outliers but can be detected and modeled.
5. **H5 — Market heterogeneity:** Revenue patterns differ between countries, requiring country-level modeling or country as a predictive feature.

## 3. Ideal Data & Rationale

* **Transaction details** — invoice ID, transaction date, amount, country, subscription type.
  *Purpose: to aggregate revenue by day/month and build time-series features such as lags and rolling averages.*
* **Customer metrics** — active users, new sign-ups, cancellations, churn rate per day/month.
  *Purpose: to test the hypothesis that customer dynamics drive revenue.*
* **Promotional events** — discount campaigns, special offers, marketing pushes.
  *Purpose: to detect and account for anomalous revenue spikes.*
* **Country information** — market category, local pricing, or macroeconomic indicators.
  *Purpose: to model market-specific differences in revenue behavior.*

## 4. Data Ingestion

A Python module (`aavail/data_ingestion.py`) was created to automatically read and clean all JSON files from the training dataset (`cs-train/`). It normalizes keys, removes non-numeric characters from invoice IDs, and parses dates for further analysis.

```python
from aavail.data_ingestion import load_aavail_data

df = load_aavail_data("cs-train")
print(df.head())
```

## 5. Exploratory Data Analysis

A first aggregation of the `price` column by month shows the revenue trend:

![Monthly Revenue](monthly_revenue.png)

* The revenue varies over time, with several spikes suggesting promotional campaigns or seasonal behavior.
* Some early months (late 2017) show small or missing values (likely incomplete data).

## 6. Findings

* The dataset contains **815k transactions** from 2017-11 to mid-2019.
* Revenue appears **highly variable with peaks**, confirming H1 (seasonality) and H4 (possible promotions/outliers).
* Time and price features are clean enough to start creating lagged features for time-series modeling in Part 2.
* Country and customer_id fields exist and can support H3/H5 testing.

**Next step:** build features (lags, rolling means) and compare time-series forecasting models (e.g., ARIMA, Holt-Winters, Prophet) for Part 2.

# IBM AI Enterprise Workflow Capstone — Part 2

## 1. Feature Engineering & Data Preparation

We aggregated the raw invoice-level data into **monthly revenue** using the `price` field. Then we created lag features to capture the autoregressive nature of the time series. The final dataset contained:

* `revenue` — total monthly revenue
* `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12` — revenue values from previous months

After cleaning and dropping rows with missing lags, we split the dataset chronologically: **80% training / 20% testing**.

```python
monthly_rev = df.groupby(df["date"].dt.to_period("M"))["price"].sum().to_timestamp().sort_index()

# Create lags
data = pd.DataFrame({"revenue": monthly_rev})
for lag in [1,2,3,6,12]:
    data[f"lag_{lag}"] = data["revenue"].shift(lag)

data = data.dropna()
train_size = int(len(data)*0.8)
train = data.iloc[:train_size]
test = data.iloc[train_size:]
```

---

## 2. Models Compared

We evaluated several **time-series forecasting models**:

* **ARIMA (3,1,1)** — classic autoregressive model
* **SARIMAX (1,1,1)(1,1,1,12)** — seasonal ARIMA
* **Holt-Winters Exponential Smoothing** — additive trend
* **Prophet** — Facebook’s open-source forecasting library

Evaluation metrics:

* **RMSE** — Root Mean Squared Error
* **MAE** — Mean Absolute Error

---

## 3. Results

| Model                       | RMSE        | MAE        |
| --------------------------- | ----------- | ---------- |
| ARIMA (3,1,1)               | ~77,579     | ~75,871    |
| SARIMAX (1,1,1)(1,1,1,12)   | ~63,387     | ~52,783    |
| **Holt-Winters (additive)** | **~13,002** | **~9,690** |
| Prophet                     | ~110,590    | ~100,905   |

**Best model:** `Holt-Winters` (additive trend) achieved the lowest RMSE and MAE.

![Best Time-Series Model](best_model.png)

---

## 4. Interpretation

* **Holt-Winters** outperformed ARIMA/SARIMAX due to the relatively short dataset and dominant trend component. Seasonality is weak and data length is insufficient for complex seasonal models.
* **Prophet** underperformed because the dataset is small and irregular.
* **RandomForest/XGBoost** were initially tested but not used for final selection since they are not pure time-series models and risk overfitting on small data.

---

## 5. Next Step — Deployment Preparation

* Retrain Holt-Winters on full data.
* Save model for serving via API:

```python
import joblib
joblib.dump(holt, "holtwinters_model.pkl")
```

* Build Flask API with endpoints `/train` and `/predict`.
* Dockerize the service for easy deployment and reproducibility.

This completes **Part 2** with a clear model selection process and performance evaluation, preparing for **Part 3: Model Deployment**.

# IBM AI Enterprise Workflow Capstone — Part 3

## 1. Goal

After selecting the **Holt-Winters** model in Part 2, the final step was to **deploy the forecasting solution** so it can be used outside a notebook.
We built a simple **REST API** that:

* Retrains the model on all available historical data (`/train` endpoint)
* Predicts revenue for any requested future month (`/predict` endpoint)
* Can be run locally with Flask and optionally containerized with Docker

---

## 2. API Architecture

We added a new folder `service/` to keep the deployment code:

```
ai-workflow-capstone/
├─ aavail/
│   └─ data_ingestion.py
├─ service/
│   ├─ app.py           # Flask REST API
│   └─ model_utils.py   # functions for training & prediction
├─ requirements.txt
└─ cs-train/            # training JSON files
```

### 2.1. Model Utilities (`service/model_utils.py`)

Handles:

* Loading and cleaning revenue data using `load_aavail_data`
* Training Holt-Winters on monthly revenue
* Saving the trained model with `joblib`
* Forecasting revenue N months ahead

```python
import joblib
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from aavail.data_ingestion import load_aavail_data

MODEL_PATH = "holtwinters_model.pkl"

def get_monthly_revenue(data_dir="cs-train"):
    df = load_aavail_data(data_dir)
    df["date"] = pd.to_datetime(dict(
        year=df["year"].astype(int),
        month=df["month"].astype(int),
        day=pd.to_numeric(df.get("day",1), errors="coerce").fillna(1).astype(int)
    ))
    return df.groupby(df["date"].dt.to_period("M"))["price"].sum().to_timestamp().sort_index()

def train_and_save(data_dir="cs-train"):
    series = get_monthly_revenue(data_dir)
    model = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
    joblib.dump(model, MODEL_PATH)
    return {"last_training_month": str(series.index.max().date()), "observations": len(series)}

def predict_for_month(target_month):
    model = joblib.load(MODEL_PATH)
    last = pd.to_datetime(model.data.dates[-1])
    target = pd.to_datetime(target_month)
    steps = (target.year - last.year) * 12 + (target.month - last.month)
    forecast = model.forecast(steps)
    return float(forecast.iloc[-1])
```

### 2.2. Flask App (`service/app.py`)

```python
from flask import Flask, request, jsonify
from model_utils import train_and_save, predict_for_month

app = Flask(__name__)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
def train():
    meta = train_and_save("cs-train")
    return jsonify({"message": "Model trained successfully", "meta": meta})

@app.post("/predict")
def predict():
    data = request.get_json(force=True)
    target = data.get("date")
    if not target:
        return {"error": "Provide 'date' in format YYYY-MM"}, 400
    try:
        forecast = predict_for_month(target)
        return {"target_month": target, "forecast": forecast}
    except FileNotFoundError:
        return {"error": "Model not trained yet. Call /train first."}, 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
```

---

## 3. Running Locally

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Start the API:**

```bash
python service/app.py
```

3. **Train & Predict:**

```bash
curl -X POST http://127.0.0.1:8000/train -H "Content-Type: application/json" -d "{}"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"date":"2019-07"}'
```

Example response:

```json
{
  "target_month": "2019-07",
  "forecast": 123456.78
}
```

---

## 4. Optional Docker Deployment

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt /app/
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/
EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "service.app:app"]
```

Build & run:

```bash
docker build -t aavail-forecast:latest .
docker run --rm -p 8000:8000 aavail-forecast:latest
```

---

## 5. Summary

* **Part 1:** Business understanding & automated data ingestion
* **Part 2:** Feature engineering, model comparison (Holt-Winters selected)
* **Part 3:** Deployment via Flask API, optional Dockerization

The project now delivers a full workflow: ingest → model → deploy → forecast via REST API.
