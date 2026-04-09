# features.py
import sys
from pathlib import Path

from config import PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from config import (
    LAG_MINUTES,
    ROLLING_WINDOWS_MINUTES
)


def _as_series(x):
    """Ensure input is a pandas Series."""
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def build_features(df):
    """
    Create modeling features from preprocessed data.
    """

    df = df.copy()

    # --------------------------------------------------
    # Define targets explicitly
    # --------------------------------------------------
    targets = {
        "h2s": "h2s_h2s_ppm",
        "nh3": "nh3_nh3_ppm"
    }
    
    # NOTE:
    # Raw east sludge flow columns are intentionally preserved.
    # A single canonical feature (east_sludge_out_gpm_combined) is created
    # for modeling, while raw signals remain for auditability and validation.

    # --------------------------------------------------
    # Resolve east sludge outflow signals robustly
    # --------------------------------------------------
    east_cols = [c for c in df.columns if c == "east_sludge_out_gpm"]

    if len(east_cols) == 1:
        df["east_sludge_out_gpm_combined"] = _as_series(df[east_cols[0]])

    elif len(east_cols) > 1:
        combined = _as_series(df[east_cols[0]])

        for col in east_cols[1:]:
            combined = combined.combine_first(_as_series(df[col]))

        df["east_sludge_out_gpm_combined"] = combined

    # --------------------------------------------------
    # Lag features (minutes)
    # --------------------------------------------------
    for lag in LAG_MINUTES:
        df[f"h2s_lag_{lag}min"] = df[targets["h2s"]].shift(lag)
        df[f"nh3_lag_{lag}min"] = df[targets["nh3"]].shift(lag)

    # NOTE ON H2S FEATURES:
    # ---------------------
    # H2S emissions are episodic and spike-driven due to localized sulfide
    # release events during biosolids handling and transfer.
    # Rolling MAX is therefore used in addition to rolling mean to capture
    # short-duration peaks that are operationally and perceptually relevant
    # for odor exposure, even if average concentrations remain low.

    # NH3 rolling max retained for symmetry and exploratory diagnostics;
    # rolling mean is the primary summary statistic for NH3 behavior.

    # --------------------------------------------------
    # Rolling window features
    # --------------------------------------------------
    for window in ROLLING_WINDOWS_MINUTES:
        df[f"h2s_roll_mean_{window}min"] = (
            df[targets["h2s"]]
            .rolling(window=window, min_periods=1)
            .mean()
        )

        df[f"h2s_roll_max_{window}min"] = (
            df[targets["h2s"]]
            .rolling(window=window, min_periods=1)
            .max()
        )

        df[f"nh3_roll_mean_{window}min"] = (
            df[targets["nh3"]]
            .rolling(window=window, min_periods=1)
            .mean()
        )

        df[f"nh3_roll_max_{window}min"] = (
            df[targets["nh3"]]
            .rolling(window=window, min_periods=1)
            .max()
        )

    # --------------------------------------------------
    # Drop rows with insufficient history
    # --------------------------------------------------
    max_lag = max(LAG_MINUTES)
    df = df.iloc[max_lag:]

    # --------------------------------------------------
    # Document derived feature groups (for clarity only)
    # --------------------------------------------------
    derived_features = [
        "east_sludge_out_gpm_combined"
    ] + [
        f"h2s_lag_{lag}min" for lag in LAG_MINUTES
    ] + [
        f"nh3_lag_{lag}min" for lag in LAG_MINUTES
    ] + [
        f"h2s_roll_mean_{w}min" for w in ROLLING_WINDOWS_MINUTES
    ] + [
        f"h2s_roll_max_{w}min" for w in ROLLING_WINDOWS_MINUTES
    ] + [
        f"nh3_roll_mean_{w}min" for w in ROLLING_WINDOWS_MINUTES
    ] + [
        f"nh3_roll_max_{w}min" for w in ROLLING_WINDOWS_MINUTES
    ]

    return df, targets, derived_features


if __name__ == "__main__":
    from scripts.load_data import load_all_data
    from scripts.preprocess import preprocess_data

    raw = load_all_data()
    clean = preprocess_data(raw)

    features, targets, derived = build_features(clean)

    print(features.head())
    print(features.info())
    print("\nDerived features:")
    print(derived)
