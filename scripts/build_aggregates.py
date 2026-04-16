from pathlib import Path
import pandas as pd

from scripts.constants import H2S, NH3, WATER_COLS as BASE_WATER_COLS
from scripts.paths import PROCESSED_DATA_DIR

DAILY_PATH = PROCESSED_DATA_DIR / "master_daily.parquet"
MONTHLY_PATH = PROCESSED_DATA_DIR / "monthly_summary.parquet"
WEEKDAY_PATH = PROCESSED_DATA_DIR / "weekday_summary.parquet"
LEGACY_BUNDLE_PATH = PROCESSED_DATA_DIR / "aggregates.parquet"

WATER_COLS = BASE_WATER_COLS + ["total_gpm", "transferred_lbs_vol_daily"]

PROCESS_COLS = [
    "ferric_available",
    "hcl_available",
    "interp_flag",
    "ferric_solution_lbs_per_day",
    "ferric_active_lbs_per_day",
]

COVERAGE_COLS = [
    "n_obs_nh3",
    "n_obs_h2s",
    "n_obs_water",
    "nh3_coverage",
    "h2s_coverage",
    "water_coverage",
]

VARIABILITY_COLS = [
    "h2s_std",
    "nh3_std",
]


def run_aggregations():
    if not DAILY_PATH.exists():
        raise FileNotFoundError("Run build_daily.py first")

    df = pd.read_parquet(DAILY_PATH).sort_index()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("master_daily.parquet must have a DatetimeIndex")

    df = df.copy()
    df["month"] = df.index.month
    df["weekday"] = df.index.dayofweek

    # --------------------------------------------------
    # Build aggregation map based on available columns
    # --------------------------------------------------
    agg_map = {}

    # Gas signals
    if NH3 in df.columns:
        agg_map[NH3] = "mean"
    if H2S in df.columns:
        agg_map[H2S] = "mean"

    # Variability
    for col in VARIABILITY_COLS:
        if col in df.columns:
            agg_map[col] = "mean"

    # Water/process operating levels
    for col in WATER_COLS:
        if col in df.columns:
            agg_map[col] = "mean"

    # Process flags / dosing
    if "ferric_available" in df.columns:
        agg_map["ferric_available"] = "mean"
    if "hcl_available" in df.columns:
        agg_map["hcl_available"] = "mean"
    if "interp_flag" in df.columns:
        agg_map["interp_flag"] = "mean"
    if "ferric_solution_lbs_per_day" in df.columns:
        agg_map["ferric_solution_lbs_per_day"] = "mean"
    if "ferric_active_lbs_per_day" in df.columns:
        agg_map["ferric_active_lbs_per_day"] = "mean"

    # Coverage / observation counts
    for col in COVERAGE_COLS:
        if col in df.columns:
            agg_map[col] = "mean"

    # --------------------------------------------------
    # Monthly summary
    # Mean of daily values within month
    # --------------------------------------------------
    monthly = (
        df.groupby("month")
        .agg(agg_map)
        .sort_index()
    )

    # Rename core columns for clarity
    monthly_rename = {}
    if NH3 in monthly.columns:
        monthly_rename[NH3] = "nh3_monthly_mean"
    if H2S in monthly.columns:
        monthly_rename[H2S] = "h2s_monthly_mean"
    if "total_gpm" in monthly.columns:
        monthly_rename["total_gpm"] = "total_gpm_monthly_mean"
    if "transferred_lbs_vol_daily" in monthly.columns:
        monthly_rename["transferred_lbs_vol_daily"] = "transferred_lbs_vol_monthly_mean"

    monthly = monthly.rename(columns=monthly_rename)

    # Number of contributing daily rows by month
    monthly["days_in_data"] = df.groupby("month").size()

    # --------------------------------------------------
    # Weekday summary
    # Mean of daily values by weekday
    # --------------------------------------------------
    weekday = (
        df.groupby("weekday")
        .agg(agg_map)
        .sort_index()
    )

    weekday_rename = {}
    if NH3 in weekday.columns:
        weekday_rename[NH3] = "nh3_weekday_mean"
    if H2S in weekday.columns:
        weekday_rename[H2S] = "h2s_weekday_mean"
    if "total_gpm" in weekday.columns:
        weekday_rename["total_gpm"] = "total_gpm_weekday_mean"
    if "transferred_lbs_vol_daily" in weekday.columns:
        weekday_rename["transferred_lbs_vol_daily"] = "transferred_lbs_vol_weekday_mean"

    weekday = weekday.rename(columns=weekday_rename)

    # Number of contributing daily rows by weekday
    weekday["days_in_data"] = df.groupby("weekday").size()

    # Optional weekday labels
    weekday["weekday_name"] = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ][: len(weekday)]

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    monthly.to_parquet(MONTHLY_PATH)
    weekday.to_parquet(WEEKDAY_PATH)

    # Legacy compatibility bundle
    bundle = {
        "monthly": monthly,
        "weekday": weekday,
    }
    pd.to_pickle(bundle, LEGACY_BUNDLE_PATH)

    print(f"✓ Monthly summary saved → {MONTHLY_PATH}")
    print(f"✓ Weekday summary saved → {WEEKDAY_PATH}")
    print(f"✓ Legacy aggregate bundle saved → {LEGACY_BUNDLE_PATH}")


if __name__ == "__main__":
    run_aggregations()
