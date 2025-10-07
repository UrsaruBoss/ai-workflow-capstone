import os
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# IMPORTANT: permite importul din aavail/
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from aavail.data_ingestion import load_aavail_data  # existent deja

MODEL_PATH = ROOT / "holtwinters_model.pkl"
META_PATH  = ROOT / "model_meta.pkl"

def get_monthly_revenue(data_dir="cs-train") -> pd.Series:
    df = load_aavail_data(data_dir)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = pd.to_numeric(df.get("day", 1), errors="coerce").fillna(1).astype(int)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["price"]
          .sum()
          .to_timestamp()
          .sort_index()
    )
    return monthly

def train_and_save(data_dir="cs-train"):
    series = get_monthly_revenue(data_dir)
    # model simplu Holt fără sezonalitate (setul e scurt)
    model = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
    joblib.dump(model, MODEL_PATH)
    meta = {
        "last_train_month": series.index.max().to_period("M").to_timestamp(),
        "n_obs": len(series)
    }
    joblib.dump(meta, META_PATH)
    return meta

def load_model():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Model or meta not found. Call /train first.")
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta

def months_between(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> int:
    """number of whole months from start (inclusive?) to end (inclusive for forecast step)"""
    ydiff = end_ts.year - start_ts.year
    mdiff = end_ts.month - start_ts.month
    return ydiff * 12 + mdiff + 1  # +1: forecast the target month itself

def predict_for_month(target_ym: str):
    """
    target_ym: 'YYYY-MM'
    returns: {'target_month': 'YYYY-MM', 'forecast': float}
    """
    model, meta = load_model()
    target = pd.Period(target_ym, freq="M").to_timestamp()
    last_train = meta["last_train_month"]
    steps = months_between(last_train, target)
    if steps <= 0:
        # dacă cere o lună din trecut, dăm valoarea observată (dacă există)
        series = model.data.endog  # numpy array
        # reconstruiem indexul temporal din model
        start = pd.date_range(
            start=model.data.dates[0],
            periods=len(series),
            freq=model.data.freq or "MS"
        )
        s = pd.Series(series, index=start)
        if target in s.index:
            return {"target_month": target.strftime("%Y-%m"), "forecast": float(s.loc[target])}
        return {"error": "Requested month is before training start."}
    # forecast n pași
    fc = model.forecast(steps=steps)
    value = float(fc.iloc[-1])
    return {"target_month": target.strftime("%Y-%m"), "forecast": value, "steps_ahead": steps}
