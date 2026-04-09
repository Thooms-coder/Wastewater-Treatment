# preprocess.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from config import (
    RESAMPLE_RULE,
    MAX_INTERP_GAP_MINUTES
)


def preprocess_data(df):
    """
    Clean and align merged sensor + operations data.
    """

    df = df.copy()

    # --------------------------------------------------
    # Ensure datetime index integrity
    # --------------------------------------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex before preprocessing.")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # --------------------------------------------------
    # Drop junk Excel export columns
    # --------------------------------------------------
    df = df.loc[:, ~df.columns.str.lower().str.startswith("unnamed")]

    df = df.drop(
        columns=[c for c in df.columns if c.lower().endswith("iso_time")],
        errors="ignore"
    )

    # --------------------------------------------------
    # Fix known naming issue safely
    # --------------------------------------------------
    if "eest_sludge_out_gpm" in df.columns:
        if "east_sludge_out_gpm" in df.columns:
            print("[WARN] Both eest_ and east_ exist. Dropping misspelled column.")
            df = df.drop(columns=["eest_sludge_out_gpm"])
        else:
            df = df.rename(columns={
                "eest_sludge_out_gpm": "east_sludge_out_gpm"
            })

    # --------------------------------------------------
    # Remove duplicate columns
    # --------------------------------------------------
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        print(f"[WARN] Dropping duplicate columns: {dups}")
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # --------------------------------------------------
    # Separate binary vs continuous columns
    # --------------------------------------------------
    binary_cols = [
        c for c in df.columns
        if c.endswith("_available")
        or c in ["sensor_valid", "truck_activity_proxy"]
    ]

    continuous_cols = [c for c in df.columns if c not in binary_cols]

    # --------------------------------------------------
    # Coerce numeric safely
    # --------------------------------------------------
    df[continuous_cols] = df[continuous_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    df[binary_cols] = df[binary_cols].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0).astype(int)

    # --------------------------------------------------
    # Resample
    # Continuous → mean
    # Binary → max (logical OR)
    # --------------------------------------------------
    df_cont = df[continuous_cols].resample(RESAMPLE_RULE).mean()
    df_bin = df[binary_cols].resample(RESAMPLE_RULE).max()

    df = pd.concat([df_cont, df_bin], axis=1)

    # --------------------------------------------------
    # Flag rows with missing data pre-interpolation
    # --------------------------------------------------
    df["interp_flag"] = df_cont.isna().any(axis=1).astype(int)

    # --------------------------------------------------
    # Controlled interpolation (continuous only)
    # --------------------------------------------------
    max_gap = f"{MAX_INTERP_GAP_MINUTES}min"
    limit = int(pd.Timedelta(max_gap) / pd.Timedelta(RESAMPLE_RULE))

    df[continuous_cols] = df[continuous_cols].interpolate(
        method="time",
        limit=limit
    )

    # Re-ensure binary integrity
    df[binary_cols] = df[binary_cols].fillna(0).astype(int)

    return df


if __name__ == "__main__":
    from scripts.load_data import load_all_data

    raw = load_all_data()
    clean = preprocess_data(raw)

    print(clean.head())
    print(clean.info())
