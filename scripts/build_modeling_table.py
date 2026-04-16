from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analytics import add_operational_features, detect_transitions
from scripts.constants import (
    EAST_GPM,
    EVENT_COLUMNS,
    FLOW,
    FLOW_COLS,
    H2S,
    NH3,
    RAW_H2S,
    RAW_NH3,
    TEMP_H2S,
    TEMP_NH3,
    WEST_GPM,
)
from scripts.paths import PROCESSED_DATA_DIR

MASTER_PATH = PROCESSED_DATA_DIR / "master_1min.parquet"
OUTPUT_PATH = PROCESSED_DATA_DIR / "modeling_table.parquet"
METADATA_PATH = PROCESSED_DATA_DIR / "modeling_table_metadata.json"

TARGET_HORIZONS_MIN = [15, 60, 180]
DEFAULT_THRESHOLD_TARGETS = {
    "nh3": 5.0,
    "h2s": 1.0,
}


def _minutes_since_event(index: pd.DatetimeIndex, event_times: pd.Index) -> pd.Series:
    if len(index) == 0:
        return pd.Series(dtype=float, index=index)

    values = np.full(len(index), np.nan, dtype=float)
    event_times = pd.DatetimeIndex(event_times).sort_values()

    if len(event_times) == 0:
        return pd.Series(values, index=index)

    event_ns = event_times.view("i8")
    idx_ns = index.view("i8")
    pos = np.searchsorted(event_ns, idx_ns, side="right") - 1
    valid = pos >= 0
    values[valid] = (idx_ns[valid] - event_ns[pos[valid]]) / 60_000_000_000

    return pd.Series(values, index=index)


def _add_transition_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for chem_name, event_col in EVENT_COLUMNS.items():
        on_events, off_events = detect_transitions(out, event_col)
        lower = chem_name.lower()

        out[f"minutes_since_{lower}_on"] = _minutes_since_event(out.index, on_events)
        out[f"minutes_since_{lower}_off"] = _minutes_since_event(out.index, off_events)

        state_change = out[event_col].fillna(0).astype(int).ne(out[event_col].fillna(0).astype(int).shift())
        state_change_times = out.index[state_change]
        out[f"minutes_since_{lower}_state_change"] = _minutes_since_event(out.index, state_change_times)

    return out


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    minutes_of_day = out.index.hour * 60 + out.index.minute

    out["hour"] = out.index.hour
    out["minute"] = out.index.minute
    out["dayofweek"] = out.index.dayofweek
    out["dayofyear"] = out.index.dayofyear
    out["month"] = out.index.month
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * minutes_of_day / 1440)
    out["hour_cos"] = np.cos(2 * np.pi * minutes_of_day / 1440)
    out["dow_sin"] = np.sin(2 * np.pi * out.index.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out.index.dayofweek / 7)

    return out


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["flow_x_hcl"] = out["total_gpm"] * out.get("hcl_available", 0)
    out["flow_x_ferric"] = out["total_gpm"] * out.get("ferric_available", 0)
    out["load_x_hcl"] = out["lbs_per_min"] * out.get("hcl_available", 0)
    out["load_x_ferric"] = out["lbs_per_min"] * out.get("ferric_available", 0)

    if TEMP_NH3 in out.columns:
        out["temp_nh3_x_hcl"] = out[TEMP_NH3] * out.get("hcl_available", 0)
        out["temp_nh3_x_ferric"] = out[TEMP_NH3] * out.get("ferric_available", 0)

    if TEMP_H2S in out.columns:
        out["temp_h2s_x_hcl"] = out[TEMP_H2S] * out.get("hcl_available", 0)
        out["temp_h2s_x_ferric"] = out[TEMP_H2S] * out.get("ferric_available", 0)

    if NH3 in out.columns:
        out["nh3_x_flow"] = out[NH3] * out["total_gpm"]
    if H2S in out.columns:
        out["h2s_x_flow"] = out[H2S] * out["total_gpm"]

    return out


def _add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for horizon in TARGET_HORIZONS_MIN:
        out[f"{NH3}_t_plus_{horizon}min"] = out[NH3].shift(-horizon)
        out[f"{H2S}_t_plus_{horizon}min"] = out[H2S].shift(-horizon)
        out[f"{RAW_NH3}_t_plus_{horizon}min"] = out[RAW_NH3].shift(-horizon) if RAW_NH3 in out.columns else np.nan
        out[f"{RAW_H2S}_t_plus_{horizon}min"] = out[RAW_H2S].shift(-horizon) if RAW_H2S in out.columns else np.nan

        out[f"nh3_delta_{horizon}min"] = out[f"{NH3}_t_plus_{horizon}min"] - out[NH3]
        out[f"h2s_delta_{horizon}min"] = out[f"{H2S}_t_plus_{horizon}min"] - out[H2S]

        out[f"nh3_exceeds_{DEFAULT_THRESHOLD_TARGETS['nh3']:.1f}_ppm_t_plus_{horizon}min"] = (
            out[f"{NH3}_t_plus_{horizon}min"] >= DEFAULT_THRESHOLD_TARGETS["nh3"]
        ).astype("Int64")
        out[f"h2s_exceeds_{DEFAULT_THRESHOLD_TARGETS['h2s']:.1f}_ppm_t_plus_{horizon}min"] = (
            out[f"{H2S}_t_plus_{horizon}min"] >= DEFAULT_THRESHOLD_TARGETS["h2s"]
        ).astype("Int64")

    return out


def _add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["gas_data_available"] = (out[[NH3, H2S]].notna().any(axis=1)).astype(int)
    out["both_gas_signals_available"] = (out[[NH3, H2S]].notna().all(axis=1)).astype(int)
    out["sensor_valid_flag"] = out.get("sensor_valid", pd.Series(0, index=out.index)).fillna(0).astype(int)

    feature_columns_required = ["total_gpm", "lbs_per_min", FLOW, WEST_GPM, EAST_GPM]
    feature_columns_present = [c for c in feature_columns_required if c in out.columns]
    if feature_columns_present:
        out["feature_row_complete"] = out[feature_columns_present].notna().all(axis=1).astype(int)
    else:
        out["feature_row_complete"] = 0

    for horizon in TARGET_HORIZONS_MIN:
        out[f"target_ready_{horizon}min"] = (
            out[[f"{NH3}_t_plus_{horizon}min", f"{H2S}_t_plus_{horizon}min"]]
            .notna()
            .all(axis=1)
            .astype(int)
        )
        out[f"model_row_valid_{horizon}min"] = (
            (out["feature_row_complete"] == 1) &
            (out["sensor_valid_flag"] == 1) &
            (out[f"target_ready_{horizon}min"] == 1)
        ).astype(int)

    return out


def _add_split_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    out["split"] = "test"
    out.iloc[:train_end, out.columns.get_loc("split")] = "train"
    out.iloc[train_end:validation_end, out.columns.get_loc("split")] = "validation"
    return out


def build_modeling_table() -> tuple[pd.DataFrame, dict]:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing master dataset: {MASTER_PATH}")

    df = pd.read_parquet(MASTER_PATH).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("master_1min.parquet must have a DatetimeIndex")

    out = add_operational_features(df)
    out["transferred_lbs_vol"] = out["lbs_per_min"]

    out = _add_transition_timing_features(out)
    out = _add_calendar_features(out)
    out = _add_interaction_features(out)
    out = _add_target_columns(out)
    out = _add_quality_flags(out)
    out = _add_split_column(out)

    metadata = {
        "source": MASTER_PATH.name,
        "output": OUTPUT_PATH.name,
        "n_rows": int(len(out)),
        "n_columns": int(len(out.columns)),
        "start": str(out.index.min()),
        "end": str(out.index.max()),
        "target_horizons_min": TARGET_HORIZONS_MIN,
        "threshold_targets": DEFAULT_THRESHOLD_TARGETS,
        "split_counts": {k: int(v) for k, v in out["split"].value_counts().sort_index().items()},
        "valid_rows_by_horizon": {
            str(h): int(out[f"model_row_valid_{h}min"].sum()) for h in TARGET_HORIZONS_MIN
        },
    }

    return out, metadata


def write_modeling_outputs(df: pd.DataFrame, metadata: dict) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    print("[build_modeling_table] Building modeling table from master dataset...")
    modeling_df, meta = build_modeling_table()
    write_modeling_outputs(modeling_df, meta)

    print(f"[build_modeling_table] Wrote {len(modeling_df):,} rows to {OUTPUT_PATH.name}")
    print(f"[build_modeling_table] Metadata written to {METADATA_PATH.name}")
    print("\nSplit counts:")
    for split_name, count in meta["split_counts"].items():
        print(f"  {split_name}: {count:,}")
    print("\nValid modeling rows by horizon:")
    for horizon, count in meta["valid_rows_by_horizon"].items():
        print(f"  {horizon} min: {count:,}")
