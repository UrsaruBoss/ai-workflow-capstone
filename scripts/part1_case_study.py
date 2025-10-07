import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aavail.data_ingestion import load_aavail_data
import matplotlib.pyplot as plt
import pandas as pd
# 1. Load data
df = load_aavail_data("cs-train")

# 2. Inspect
print("\n=== FIRST ROWS ===")
print(df.head())
print("\n=== INFO ===")
print(df.info())

# 3. Simple plot if we have date & price/revenue column
possible_cols = [c for c in df.columns if "price" in c.lower() or "revenue" in c.lower() or "amount" in c.lower()]
# we don't have a clean 'date' column, but we have 'year','month','day'
if {"year", "month"}.issubset(df.columns) and "price" in df.columns:
    # convert year/month/day to datetime
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = pd.to_numeric(df.get("day", 1), errors="coerce").fillna(1).astype(int)

    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    monthly = df.groupby(df["date"].dt.to_period("M"))["price"].sum()
    monthly.plot(title="Monthly Revenue (sum of price)")
    plt.ylabel("Revenue")
    plt.show()
else:
    print("Could not construct date or price column for plotting")
