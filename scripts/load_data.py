# load_data.py

import sys
from pathlib import Path
import pandas as pd

from config import PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT))

from scripts.paths import RAW_DATA_DIR


# --------------------------------------------------
# Utility: normalize column names
# --------------------------------------------------
def normalize_columns(df):
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    return df


# --------------------------------------------------
# Utility: safe datetime parsing
# --------------------------------------------------
def parse_datetime(series):
    """
    Robust datetime parsing supporting:
    - 11/10/25 12:01:00 AM  (plant export)
    - 9-12-2025, 7-22-54 AM (gas CSV)
    - 2025-11-10 00:01:00   (ISO)
    """

    s = series.astype(str).str.strip()

    # 1️⃣ Plant format
    dt = pd.to_datetime(
        s,
        format="%m/%d/%y %I:%M:%S %p",
        errors="coerce"
    )

    # 2️⃣ Gas CSV format (only where needed)
    mask = dt.isna()
    if mask.any():
        dt2 = pd.to_datetime(
            s[mask],
            format="%m-%d-%Y, %I-%M-%S %p",
            errors="coerce"
        )
        dt.loc[mask] = dt2

    # 3️⃣ Final fallback (ISO or other valid strings)
    mask = dt.isna()
    if mask.any():
        dt3 = pd.to_datetime(
            s[mask],
            errors="coerce"
        )
        dt.loc[mask] = dt3

    return dt

# --------------------------------------------------
# Utility: remove duplicate columns safely
# --------------------------------------------------
def remove_duplicate_columns(df, label=""):
    dups = df.columns[df.columns.duplicated()].tolist()
    if dups:
        print(f"[WARN] Dropping duplicate columns in {label}: {dups}")
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def detect_water_header_row(raw):
    """
    Some plant exports contain a duplicated two-row header where the first row has
    a generic label like "Digesters Sludge Out Flow" and the second row contains
    the real West/East flow column names. Prefer the row with the strongest match
    to the expected operational headers.
    """
    header_candidates = []
    expected_terms = [
        "west sludge out",
        "east sludge out",
        "eest sludge out",
        "digesters sludge out flow",
        "gbt sludge feed pump",
    ]

    for i in range(min(len(raw), 12)):
        values = raw.iloc[i].astype(str).str.strip()
        if not values.str.contains("Time", case=False).any():
            continue

        lowered = values.str.lower()
        score = sum(lowered.str.contains(term, regex=False).sum() for term in expected_terms)
        unnamed_penalty = lowered.str.startswith("unnamed").sum()
        header_candidates.append((score, -unnamed_penalty, i))

    if not header_candidates:
        return None

    header_candidates.sort(reverse=True)
    return header_candidates[0][2]


# --------------------------------------------------
# H2S Loader
# --------------------------------------------------
def load_h2s_data():
    files = sorted(RAW_DATA_DIR.glob("*H2S*.csv"))
    if not files:
        raise ValueError("No H2S CSV files found.")

    dfs = []

    for file in files:
        df = pd.read_csv(file, skiprows=8, sep=",", engine="python")

        df["Time Stamp"] = parse_datetime(df["Time Stamp"])
        df = df.dropna(subset=["Time Stamp"])

        if df.empty:
            raise ValueError(f"All timestamps invalid in {file.name}")

        df = df.set_index("Time Stamp")
        df = normalize_columns(df)
        df = df.add_prefix("h2s_")

        dfs.append(df)

    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = remove_duplicate_columns(combined, "H2S")

    return combined


# --------------------------------------------------
# NH3 Loader
# --------------------------------------------------
def load_nh3_data():
    files = sorted(RAW_DATA_DIR.glob("*NH3*.csv"))
    if not files:
        raise ValueError("No NH3 CSV files found.")

    dfs = []

    for file in files:
        df = pd.read_csv(file, skiprows=8, sep=",", engine="python")

        df["Time Stamp"] = parse_datetime(df["Time Stamp"])
        df = df.dropna(subset=["Time Stamp"])

        if df.empty:
            raise ValueError(f"All timestamps invalid in {file.name}")

        df = df.set_index("Time Stamp")
        df = normalize_columns(df)
        df = df.add_prefix("nh3_")

        dfs.append(df)

    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = remove_duplicate_columns(combined, "NH3")

    return combined


# --------------------------------------------------
# Water Reclamation Loader
# --------------------------------------------------
def load_water_reclamation_data():
    files = sorted(RAW_DATA_DIR.glob("Water Reclamation*.xlsx"))
    if not files:
        raise ValueError("No water reclamation Excel files found.")

    dfs = []

    for file in files:

        # Read raw without header
        raw = pd.read_excel(file, header=None)

        # Find the strongest header row containing the real flow labels.
        header_row = detect_water_header_row(raw)

        if header_row is None:
            raise ValueError(f"Could not find header row in {file.name}")

        # Re-read using detected header row
        df = pd.read_excel(file, header=header_row)

        # Drop completely empty rows
        df = df.dropna(how="all")

        # Normalize column names
        df = normalize_columns(df)

        if "time" not in df.columns:
            raise ValueError(f"'Time' column missing after normalization in {file.name}")

        # Parse datetime
        df["time"] = parse_datetime(df["time"])
        df = df.dropna(subset=["time"]).set_index("time")

        dfs.append(df)

    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    combined = remove_duplicate_columns(combined, "Water")

    return combined


# --------------------------------------------------
# Master Merge
# --------------------------------------------------
def load_all_data():
    h2s = load_h2s_data()
    nh3 = load_nh3_data()
    water = load_water_reclamation_data()

    df = h2s.join(nh3, how="outer")
    df = df.join(water, how="outer")

    df = df.sort_index()
    df = remove_duplicate_columns(df, "MASTER")

    # Guard against NaT index
    if df.index.isna().any():
        raise ValueError("NaT detected in master index.")

    if df.index.min() is pd.NaT or df.index.max() is pd.NaT:
        raise ValueError("Invalid index range detected.")

    return df


# --------------------------------------------------
# Debug
# --------------------------------------------------
if __name__ == "__main__":
    df = load_all_data()
    print(df.head())
    print(df.info())
