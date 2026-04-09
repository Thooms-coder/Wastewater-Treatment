import numpy as np
import pandas as pd

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
NH3 = "nh3_roll_mean_15min"
H2S = "h2s_roll_max_15min"

BASELINE_WINDOW = (-48 * 60, -12 * 60)
POST_WINDOW = (12 * 60, 96 * 60)


# --------------------------------------------------
# INTERNAL HELPERS
# --------------------------------------------------
def _validate_column(df, col):
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")


def _safe_series(df, col):
    if col not in df.columns:
        return pd.Series(dtype=float)
    return df[col].dropna()


# --------------------------------------------------
# EVENT DETECTION
# --------------------------------------------------
def detect_transitions(df, column):
    if column not in df.columns:
        return pd.Index([]), pd.Index([])

    diff = df[column].diff()

    on_events = df.index[diff == 1]
    off_events = df.index[diff == -1]

    return on_events, off_events


def detect_all_transitions(df, event_columns):
    events = {}

    for name, col in event_columns.items():
        if col not in df.columns:
            continue

        diff = df[col].diff()

        events[f"{name}_ON"] = list(df.index[diff == 1])
        events[f"{name}_OFF"] = list(df.index[diff == -1])

    return events


# --------------------------------------------------
# EVENT METRICS CORE
# --------------------------------------------------
def window_slice(series, window):
    if series is None or series.empty:
        return pd.Series(dtype=float)

    return series.loc[
        (series.index >= window[0]) &
        (series.index <= window[1])
    ]


def compute_single_event_metrics(baseline, post):
    if baseline.empty or post.empty:
        return None

    baseline_val = baseline.median()
    post_val = post.median()

    delta = post_val - baseline_val

    if pd.isna(baseline_val) or baseline_val == 0:
        pct_change = np.nan
    else:
        pct_change = (delta / baseline_val) * 100

    # Time to minimum (useful for suppression effects)
    time_to_min = np.nan if post.empty else post.idxmin()

    # Persistence below baseline
    below = post[post < baseline_val]
    persistence = (
        0 if below.empty else below.index.max() - below.index.min()
    )

    # Variability (robust spread)
    iqr_post = post.quantile(0.75) - post.quantile(0.25)

    return {
        "baseline": baseline_val,
        "post": post_val,
        "delta": delta,
        "percent_change": pct_change,
        "time_to_min": time_to_min,
        "persistence": persistence,
        "post_iqr": iqr_post,
    }


def aggregate_event_metrics(metrics_list):
    if not metrics_list:
        return {}

    df = pd.DataFrame(metrics_list)

    return {
        "baseline": df["baseline"].median(),
        "post": df["post"].median(),
        "delta": df["delta"].median(),
        "percent_change": df["percent_change"].median(),
        "time_to_min": df["time_to_min"].median(),
        "persistence": df["persistence"].median(),
        "post_iqr": df["post_iqr"].median(),
        "n_events": len(df),
    }


def compute_event_metrics(df, event_col, signal_col):
    if event_col not in df.columns or signal_col not in df.columns:
        return pd.DataFrame()

    on_events, off_events = detect_transitions(df, event_col)

    results = []

    for event_type, events in {"ON": on_events, "OFF": off_events}.items():
        event_metrics = []

        for t in events:
            s = _safe_series(df, signal_col)
            if s.empty:
                continue

            # Align time relative to event
            rel_time = (s.index - t).total_seconds() / 60
            s_aligned = s.copy()
            s_aligned.index = rel_time.astype(int)

            baseline = window_slice(s_aligned, BASELINE_WINDOW)
            post = window_slice(s_aligned, POST_WINDOW)

            metrics = compute_single_event_metrics(baseline, post)
            if metrics is None:
                continue

            event_metrics.append(metrics)

        if event_metrics:
            agg = aggregate_event_metrics(event_metrics)
            agg["event_type"] = event_type
            results.append(agg)

    return pd.DataFrame(results)


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
def add_operational_features(df):
    if df is None or df.empty:
        return df

    df = df.copy()

    # Ensure total flow exists
    flow_cols = [
        "west_sludge_out_gpm",
        "east_sludge_out_gpm",
        "digesters_sludge_out_flow",
    ]

    for col in flow_cols:
        if col not in df.columns:
            df[col] = 0.0

    df["total_gpm"] = df[flow_cols].sum(axis=1)

    # Mass transfer approximation
    k = 8.34 * 1.38 * 0.66
    df["lbs_per_min"] = df["total_gpm"] * k

    # Normalize gas by load
    eps = 1e-9

    if NH3 in df.columns:
        df["nh3_per_lb"] = df[NH3] / (df["lbs_per_min"] + eps)

    if H2S in df.columns:
        df["h2s_per_lb"] = df[H2S] / (df["lbs_per_min"] + eps)

    return df


# --------------------------------------------------
# ANOMALY DETECTION
# --------------------------------------------------
def add_zscore(df, col, window=1440):
    _validate_column(df, col)

    rolling_mean = df[col].rolling(window, min_periods=window // 10).mean()
    rolling_std = df[col].rolling(window, min_periods=window // 10).std()

    z = (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)

    return z


def detect_anomalies(df, col, threshold=3):
    if col not in df.columns:
        return pd.DataFrame()

    z = add_zscore(df, col)

    anomalies = df[z > threshold].copy()
    anomalies["z_score"] = z[z > threshold]

    return anomalies