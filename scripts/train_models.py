from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.constants import H2S, NH3, RAW_H2S, RAW_NH3, TEMP_H2S, TEMP_NH3
from scripts.paths import PROCESSED_DATA_DIR

MODELING_TABLE_PATH = PROCESSED_DATA_DIR / "modeling_table.parquet"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
METADATA_DIR = MODELS_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR = MODELS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CONFIGS = {
    "nh3_t_plus_60min": {
        "target_col": f"{NH3}_t_plus_60min",
        "delta_col": "nh3_delta_60min",
        "current_col": NH3,
        "threshold_col": "nh3_exceeds_5.0_ppm_t_plus_60min",
        "valid_col": "model_row_valid_60min",
    },
    "h2s_t_plus_60min": {
        "target_col": f"{H2S}_t_plus_60min",
        "delta_col": "h2s_delta_60min",
        "current_col": H2S,
        "threshold_col": "h2s_exceeds_1.0_ppm_t_plus_60min",
        "valid_col": "model_row_valid_60min",
    },
}

# Modeling approach (chosen empirically — see the training verdict):
# Predict the CHANGE y(t+H) - y(t) from a pruned, physically-meaningful feature
# set, then reconstruct the level as y(t) + alpha * predicted_change, with the
# shrinkage alpha tuned on validation. This anchors the forecast to the current
# value like persistence (so it can't blow up in a new regime the way a
# 190-feature level model did), while still exploiting short-horizon structure.
FEATURE_EXCLUDE_PATTERNS = ["_t_plus_", "_delta_", "_exceeds_", "target_ready_", "model_row_valid_"]
FEATURE_KEEP_EXACT = {
    "total_gpm", "lbs_per_min", "transferred_lbs_vol",
    "west_sludge_out_gpm", "east_sludge_out_gpm", "east_sludge_out_gpm_combined",
    "ferric_available", "hcl_available",
    "ferric_active_lbs_per_day", "hcl_active_lbs_per_day",
    "ferric_solution_lbs_per_day", "hcl_solution_lbs_per_day",
    "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "hour", "dayofweek", "month",
}
FEATURE_KEEP_PREFIX = (
    "nh3_lag_", "h2s_lag_", "nh3_roll_", "h2s_roll_", "minutes_since_",
    "flow_x_", "load_x_", "temp_nh3_x_", "temp_h2s_x_", "nh3_x_", "h2s_x_",
)
FEATURE_KEEP_CONTAINS = ("temperature", "humidity")


def build_feature_columns(df: pd.DataFrame) -> list[str]:
    """Pruned, leakage-free, physically-meaningful features (drops the ~170
    daily-report columns that made the level model overfit)."""
    out = []
    for c in df.columns:
        if any(p in c for p in FEATURE_EXCLUDE_PATTERNS) or c == "split":
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if c in FEATURE_KEEP_EXACT or c.startswith(FEATURE_KEEP_PREFIX) or any(k in c for k in FEATURE_KEEP_CONTAINS):
            out.append(c)
    return sorted(set(out))


def _best_shrinkage(y_true, current, delta_pred):
    """alpha in [0,1] minimizing RMSE(y_true, current + alpha*delta_pred).
    alpha=0 recovers persistence; alpha=1 trusts the model fully."""
    alphas = np.linspace(0.0, 1.0, 21)
    return float(min(alphas, key=lambda a: np.sqrt(mean_squared_error(y_true, current + a * delta_pred))))


def metric_dict(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def threshold_metrics(y_true: pd.Series, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    true_flag = (y_true >= threshold).astype(int)
    pred_flag = (pd.Series(y_pred, index=y_true.index) >= threshold).astype(int)

    accuracy = float((true_flag == pred_flag).mean())
    tp = int(((true_flag == 1) & (pred_flag == 1)).sum())
    fp = int(((true_flag == 0) & (pred_flag == 1)).sum())
    fn = int(((true_flag == 1) & (pred_flag == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": float(precision) if not np.isnan(precision) else np.nan,
        "recall": float(recall) if not np.isnan(recall) else np.nan,
    }


def persistence_baseline(y_current: pd.Series, fallback_value: float) -> np.ndarray:
    filled = y_current.copy().fillna(fallback_value)
    return filled.to_numpy(copy=True)


def load_modeling_table() -> pd.DataFrame:
    if not MODELING_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing modeling table: {MODELING_TABLE_PATH}. Run scripts.build_modeling_table first."
        )
    df = pd.read_parquet(MODELING_TABLE_PATH).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("modeling_table.parquet must have a DatetimeIndex")
    return df


def assign_valid_splits(df: pd.DataFrame, embargo_minutes: int = 180) -> pd.DataFrame:
    out = df.sort_index().copy()
    n = len(out)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    out["model_split"] = "test"
    col = out.columns.get_loc("model_split")
    out.iloc[:train_end, col] = "train"
    out.iloc[train_end:validation_end, col] = "validation"

    # Embargo rows straddling the chronological boundaries: with forward-shifted
    # targets (up to `embargo_minutes` ahead), the tail of each block would
    # otherwise carry targets that live inside the next block. Excluding them
    # prevents future information from leaking across the split.
    if n and embargo_minutes:
        emb = pd.Timedelta(minutes=embargo_minutes)
        if train_end > 0:
            train_last = out.index[train_end - 1]
            out.loc[(out["model_split"] == "train") & (out.index > train_last - emb), "model_split"] = "embargo"
        if validation_end > train_end:
            val_last = out.index[validation_end - 1]
            out.loc[(out["model_split"] == "validation") & (out.index > val_last - emb), "model_split"] = "embargo"
    return out


def train_target_model(
    df: pd.DataFrame,
    target_name: str,
    target_cfg: dict,
    feature_cols: list[str],
) -> dict:
    target_col, delta_col, current_col = target_cfg["target_col"], target_cfg["delta_col"], target_cfg["current_col"]
    valid_mask = (df[target_cfg["valid_col"]] == 1) & df[target_col].notna() & df[delta_col].notna()
    model_df = assign_valid_splits(df.loc[valid_mask].copy())

    train_df = model_df[model_df["model_split"] == "train"]
    validation_df = model_df[model_df["model_split"] == "validation"]
    test_df = model_df[model_df["model_split"] == "test"]

    # Train on the CHANGE (delta), not the level.
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_depth=6,
        min_samples_leaf=50,
        l2_regularization=0.1,
        early_stopping=False,
        random_state=42,
    )
    model.fit(train_df[feature_cols], train_df[delta_col])

    # Shrinkage alpha tuned on validation: level = current + alpha * delta_pred.
    val_delta_pred = model.predict(validation_df[feature_cols])
    alpha = _best_shrinkage(
        validation_df[target_col].to_numpy(),
        validation_df[current_col].to_numpy(),
        val_delta_pred,
    )
    baseline_fallback = float(train_df[target_col].median())

    outputs = {
        "target_name": target_name,
        "target_col": target_col,
        "approach": "delta + pruned features + validation-tuned shrinkage",
        "shrinkage_alpha": alpha,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "splits": {},
    }

    predictions_frames = []
    threshold = None
    if "threshold_col" in target_cfg and target_cfg["threshold_col"] in model_df.columns:
        threshold_label = target_cfg["threshold_col"]
        threshold = float(threshold_label.split("_")[2])

    for split_name, split_df in [("train", train_df), ("validation", validation_df), ("test", test_df)]:
        X = split_df[feature_cols]
        y = split_df[target_col]
        # Reconstruct the level from current value + shrunk predicted change.
        y_pred = split_df[current_col].to_numpy() + alpha * model.predict(X)

        baseline_pred = persistence_baseline(split_df[current_col], baseline_fallback)
        model_metrics = metric_dict(y, y_pred)
        baseline_metrics = metric_dict(y, baseline_pred)

        split_summary = {
            "n_rows": int(len(split_df)),
            "model_metrics": model_metrics,
            "baseline_metrics": baseline_metrics,
        }

        if threshold is not None:
            split_summary["model_threshold_metrics"] = threshold_metrics(y, y_pred, threshold)
            split_summary["baseline_threshold_metrics"] = threshold_metrics(y, baseline_pred, threshold)

        outputs["splits"][split_name] = split_summary

        pred_frame = pd.DataFrame(
            {
                "timestamp": split_df.index,
                "split": split_name,
                "target_name": target_name,
                "y_true": y.to_numpy(),
                "y_pred_model": y_pred,
                "y_pred_persistence": baseline_pred,
            }
        )
        predictions_frames.append(pred_frame)

    model_path = MODELS_DIR / f"{target_name}_hist_gbr.joblib"
    metrics_path = METADATA_DIR / f"{target_name}_metrics.json"
    predictions_path = PREDICTIONS_DIR / f"{target_name}_predictions.parquet"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(outputs, indent=2))
    pd.concat(predictions_frames, ignore_index=True).to_parquet(predictions_path)

    # Honest verdict: does the trained model actually beat naive persistence on
    # the held-out (chronological) test set? If not, persistence is the
    # recommended predictor and the GBR is experimental only.
    test = outputs["splits"].get("test", {})
    model_rmse = test.get("model_metrics", {}).get("rmse")
    base_rmse = test.get("baseline_metrics", {}).get("rmse")
    beats = model_rmse is not None and base_rmse is not None and model_rmse < base_rmse
    outputs["beats_persistence_on_test"] = bool(beats)
    outputs["recommended_predictor"] = "model" if beats else "persistence"

    outputs["model_path"] = str(model_path)
    outputs["metrics_path"] = str(metrics_path)
    outputs["predictions_path"] = str(predictions_path)
    return outputs


def train_all_models() -> dict:
    df = load_modeling_table()
    feature_cols = build_feature_columns(df)

    summary = {
        "source": str(MODELING_TABLE_PATH),
        "n_total_rows": int(len(df)),
        "n_feature_columns": len(feature_cols),
        "feature_columns_preview": feature_cols[:25],
        "targets": {},
    }

    for target_name, target_cfg in TARGET_CONFIGS.items():
        summary["targets"][target_name] = train_target_model(df, target_name, target_cfg, feature_cols)

    summary_path = METADATA_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    return summary


if __name__ == "__main__":
    print("[train_models] Training baseline HistGradientBoosting models...")
    result = train_all_models()
    print(f"[train_models] Wrote training summary to {result['summary_path']}")
    for target_name, info in result["targets"].items():
        print(f"\nTarget: {target_name}")
        for split_name in ["validation", "test"]:
            split_metrics = info["splits"][split_name]
            model_rmse = split_metrics["model_metrics"]["rmse"]
            base_rmse = split_metrics["baseline_metrics"]["rmse"]
            model_r2 = split_metrics["model_metrics"]["r2"]
            base_r2 = split_metrics["baseline_metrics"]["r2"]
            print(
                f"  {split_name}: model rmse={model_rmse:.4f}, baseline rmse={base_rmse:.4f}, "
                f"model r2={model_r2:.4f}, baseline r2={base_r2:.4f}"
            )
        verdict = (
            "MODEL beats persistence"
            if info["recommended_predictor"] == "model"
            else "persistence WINS on test — GBR is experimental, not recommended"
        )
        print(f"  verdict: {verdict}")
