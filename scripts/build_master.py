# scripts/build_master.py

import sys
from pathlib import Path
from config import PROJECT_ROOT

sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from scripts.load_data import load_all_data
from scripts.preprocess import preprocess_data
from scripts.features import build_features
from scripts.events import add_event_flags
from scripts.chemistry_features import build_chemistry_features
from config import PROCESSED_DATA_DIR

def build_hourly_table(df):
    """
    EXACT match to full_timeseries_plots.py logic
    """

    df = df.copy()

    WEST = "west_sludge_out_gpm"
    EAST = "east_sludge_out_gpm"
    DIG = "digesters_sludge_out_flow"

    # -----------------------------
    # Match full_timeseries logic
    # -----------------------------
    for col in [WEST, EAST, DIG]:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0)
        else:
            df[col] = 0

    df["total_gpm"] = df[WEST] + df[EAST] + df[DIG]

    k = 8.34 * 1.38 * 0.66

    df["lbs_per_min"] = df["total_gpm"] * k

    # -----------------------------
    # Hourly aggregation (IDENTICAL)
    # -----------------------------
    hourly = pd.DataFrame()

    hourly["flow_gal_hr"] = (
        df["total_gpm"]
        .resample("1h", label="right", closed="right")
        .sum()
    )

    hourly["lbs_volatile"] = (
        df["lbs_per_min"]
        .resample("1h", label="right", closed="right")
        .sum()
    )

    hourly["fecl3_lbs"] = hourly["lbs_volatile"] / 24.3

    # -----------------------------
    # Gas signals (same bins)
    # -----------------------------
    agg = {}

    if "h2s_roll_max_15min" in df.columns:
        agg["h2s_roll_max_15min"] = "max"

    if "nh3_roll_mean_15min" in df.columns:
        agg["nh3_roll_mean_15min"] = "mean"

    if agg:
        gas = df.resample("1h", label="right", closed="right").agg(agg)
        hourly = hourly.join(gas)

    return hourly

def build_master_table():
    """
    Build the full master dataset used for exploration and modeling.
    Preserves full 1-minute water timeline while integrating gas data.
    """

    # --------------------------------------------------
    # Load and preprocess
    # --------------------------------------------------
    raw = load_all_data()
    clean = preprocess_data(raw)

    # --------------------------------------------------
    # REMOVE DUPLICATE COLUMNS (CRITICAL FIX)
    # --------------------------------------------------
    if clean.columns.duplicated().any():
        dupes = clean.columns[clean.columns.duplicated()].tolist()
        print(f"Removing duplicate columns early: {dupes}")
        clean = clean.loc[:, ~clean.columns.duplicated()]

    # --------------------------------------------------
    # Enforce continuous 1-minute timeline (CRITICAL FIX)
    # --------------------------------------------------
    full_index = pd.date_range(
        start=clean.index.min(),
        end=clean.index.max(),
        freq="1min"
    )
    clean = clean.reindex(full_index)

    # Forward-fill WATER data (flow, tanks, pumps, etc.)
    water_cols = [c for c in clean.columns if "sludge" in c or "flow" in c or "pump" in c]
    
    print("Number of columns:", len(clean.columns))
    print("Unique columns:", len(clean.columns.unique()))
    dupes = clean.columns[clean.columns.duplicated()].tolist()
    print("Duplicate columns:", dupes)

    clean[water_cols] = clean[water_cols].ffill()

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------
    features_df, targets, derived_features = build_features(clean)

    # --------------------------------------------------
    # Add operational events
    # --------------------------------------------------
    with_events = add_event_flags(features_df)

    # --------------------------------------------------
    # Add chemistry-informed features
    # --------------------------------------------------
    master = build_chemistry_features(with_events)

    # --------------------------------------------------
    # DO NOT DROP WATER-ONLY ROWS
    # (Sensor validity should not delete flow data)
    # --------------------------------------------------
    if "sensor_valid" in master.columns:
        master["sensor_valid"] = master["sensor_valid"].fillna(0)

    # --------------------------------------------------
    # Sort and finalize
    # --------------------------------------------------
    master = master.sort_index()

    # --------------------------------------------------
    # Enforce unique column names
    # --------------------------------------------------
    if master.columns.duplicated().any():
        dupes = master.columns[master.columns.duplicated()].unique().tolist()
        print(f"Resolving duplicate columns before save: {dupes}")

        if "east_sludge_out_gpm" in dupes:
            master = master.drop(columns=["east_sludge_out_gpm"])

        assert not master.columns.duplicated().any(), "Duplicate columns remain"

    metadata = {
        "targets": targets,
        "derived_features": derived_features,
        "n_rows": len(master),
        "start": str(master.index.min()),
        "end": str(master.index.max()),
    }

    return master, metadata


if __name__ == "__main__":
    
    print("[build_master] Starting master dataset build…")

    master, meta = build_master_table()

    # -----------------------------
    # NEW: Build hourly dataset
    # -----------------------------
    hourly = build_hourly_table(master)

    hourly_path_parquet = PROCESSED_DATA_DIR / "master_1h.parquet"
    hourly_path_excel = PROCESSED_DATA_DIR / "master_1h.xlsx"

    hourly.to_parquet(hourly_path_parquet)
    hourly.to_excel(hourly_path_excel)

    print(f"[build_master] Wrote hourly dataset: {hourly_path_excel.name}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_DIR / "master_1min.parquet"
    master.to_parquet(output_path)

    print(f"[build_master] Wrote {len(master):,} rows to {output_path.name}")

    print("\nMaster dataset written to:")
    print(output_path)

    print("\nMetadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    print("\nPreview:")
    print(master.head())

    print("\nInfo:")
    print(master.info())
