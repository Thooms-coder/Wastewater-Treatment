from pathlib import Path
import pandas as pd

from scripts.constants import EAST_GPM, FLOW, FLOW_COLS, H2S, NH3, WATER_COLS, WEST_GPM
from scripts.paths import PROCESSED_DATA_DIR

MASTER_PATH = PROCESSED_DATA_DIR / "master_1min.parquet"
OUTPUT_PATH = PROCESSED_DATA_DIR / "master_daily.parquet"

OPTIONAL_PROCESS_COLS = [
    "ferric_available",
    "hcl_available",
    "interp_flag",
    "ferric_solution_lbs_per_day",
    "ferric_active_lbs_per_day",
]


def run_daily_aggregation():
    if not MASTER_PATH.exists():
        raise FileNotFoundError("Run build_master.py first")

    df = pd.read_parquet(MASTER_PATH).sort_index()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("master_1min.parquet must have a DatetimeIndex")

    agg_map = {}

    # --------------------------------------------------
    # Gas signals
    # NH3 = continuous -> mean
    # H2S = spike-driven -> max
    # --------------------------------------------------
    if NH3 in df.columns:
        agg_map[NH3] = "mean"

    if H2S in df.columns:
        agg_map[H2S] = "max"

    # --------------------------------------------------
    # Water/process flow columns
    # Daily mean preserves operating level
    # --------------------------------------------------
    for col in WATER_COLS:
        if col in df.columns:
            agg_map[col] = "mean"

    # --------------------------------------------------
    # Optional process/binary columns
    # availability flags -> max (did it occur that day?)
    # interp_flag -> sum (how many interpolated rows)
    # ferric dosing -> mean daily operating level
    # --------------------------------------------------
    if "ferric_available" in df.columns:
        agg_map["ferric_available"] = "max"

    if "hcl_available" in df.columns:
        agg_map["hcl_available"] = "max"

    if "interp_flag" in df.columns:
        agg_map["interp_flag"] = "sum"

    if "ferric_solution_lbs_per_day" in df.columns:
        agg_map["ferric_solution_lbs_per_day"] = "mean"

    if "ferric_active_lbs_per_day" in df.columns:
        agg_map["ferric_active_lbs_per_day"] = "mean"

    # --------------------------------------------------
    # Base daily aggregation
    # --------------------------------------------------
    daily = df.resample("1D").agg(agg_map)

    # --------------------------------------------------
    # Gas variability metrics
    # --------------------------------------------------
    if H2S in df.columns:
        daily["h2s_std"] = df[H2S].resample("1D").std()

    if NH3 in df.columns:
        daily["nh3_std"] = df[NH3].resample("1D").std()

    # --------------------------------------------------
    # Observation counts / coverage
    # Keep separate because water and gas coverage differ
    # --------------------------------------------------
    if NH3 in df.columns:
        daily["n_obs_nh3"] = df[NH3].resample("1D").count()
        daily["nh3_coverage"] = daily["n_obs_nh3"] / 1440

    if H2S in df.columns:
        daily["n_obs_h2s"] = df[H2S].resample("1D").count()
        daily["h2s_coverage"] = daily["n_obs_h2s"] / 1440

    water_count_source = None
    for col in [FLOW, EAST_GPM, WEST_GPM]:
        if col in df.columns:
            water_count_source = col
            break

    if water_count_source is not None:
        daily["n_obs_water"] = df[water_count_source].resample("1D").count()
        daily["water_coverage"] = daily["n_obs_water"] / 1440

    # --------------------------------------------------
    # Derived daily totals
    # Match your existing process logic where possible
    # --------------------------------------------------
    available_flow_cols = [c for c in WATER_COLS if c in df.columns]

    if available_flow_cols:
        total_gpm = pd.Series(0.0, index=df.index)

        for col in FLOW_COLS:
            if col in df.columns:
                total_gpm = total_gpm.add(df[col].fillna(0), fill_value=0)

        df["total_gpm_daily_helper"] = total_gpm

        # Daily mean total GPM
        daily["total_gpm"] = df["total_gpm_daily_helper"].resample("1D").mean()

        # Volatile transfer logic carried from your existing timeline scripts
        k = 8.34 * 1.38 * 0.66
        df["lbs_per_min_daily_helper"] = df["total_gpm_daily_helper"] * k

        # Daily transferred volatile lbs = sum of minute-level lbs/min
        daily["transferred_lbs_vol_daily"] = (
            df["lbs_per_min_daily_helper"]
            .resample("1D")
            .sum()
        )

    # --------------------------------------------------
    # Keep days that have any meaningful data
    # Do not drop water-only days just because gas is missing
    # --------------------------------------------------
    coverage_cols = [
        c for c in ["n_obs_nh3", "n_obs_h2s", "n_obs_water"] if c in daily.columns
    ]

    if coverage_cols:
        keep_mask = daily[coverage_cols].fillna(0).sum(axis=1) > 0
        daily = daily.loc[keep_mask]
    else:
        daily = daily.dropna(how="all")

    daily.to_parquet(OUTPUT_PATH)
    print(f"✓ Daily dataset saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    run_daily_aggregation()
