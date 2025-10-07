import pandas as pd
import json
from pathlib import Path

def load_aavail_data(data_dir: str = "cs-train") -> pd.DataFrame:
    """
    Reads all JSON files from the cs-train directory,
    normalizes them and returns a single Pandas DataFrame.
    """
    all_frames = []

    for file in Path(data_dir).glob("*.json"):
        print(f"Reading {file.name}")
        with open(file, "r") as f:
            data = json.load(f)

        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.json_normalize(data)
        else:
            df = pd.json_normalize(data, sep="_")

        # Clean invoice ids (remove letters)
        if "invoice" in df.columns:
            df["invoice"] = df["invoice"].astype(str).str.replace(r"[^0-9]", "", regex=True)

        # Parse date if exists
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        all_frames.append(df)

    if not all_frames:
        raise ValueError("⚠️ No JSON files found in the cs-train folder.")

    combined = pd.concat(all_frames, ignore_index=True)
    return combined
