import numpy as np
import pandas as pd
from scripts.constants import (
    BASELINE_WINDOW,
    EVENT_COLUMNS as DEFAULT_EVENT_COLUMNS,
    FLOW_COLS,
    H2S,
    NH3,
    POST_WINDOW,
    PRETREND_TOL as DEFAULT_PRETREND_TOL,
    PRETREND_WINDOW as DEFAULT_PRETREND_WINDOW,
)


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

    series = df[column].fillna(0).astype(int)
    diff = series.diff()

    on_events = df.index[diff == 1]
    off_events = df.index[diff == -1]

    return on_events, off_events


def detect_all_transitions(df, event_columns=None):
    events = {}

    if event_columns is None:
        event_columns = DEFAULT_EVENT_COLUMNS

    for name, col in event_columns.items():
        if col not in df.columns:
            continue

        on_events, off_events = detect_transitions(df, col)
        events[f"{name}_ON"] = list(on_events)
        events[f"{name}_OFF"] = list(off_events)

    return events


# --------------------------------------------------
# EVENT WINDOW HELPERS
# --------------------------------------------------
def extract_relative_series(df, event_time, column):
    if column not in df.columns:
        return None

    series = df[column].dropna().copy()
    if series.empty:
        return None

    rel_time = (series.index - event_time).total_seconds() / 60
    series.index = rel_time.astype(int)
    return series


def extract_event_window(df, event_time, column, window_minutes):
    if column not in df.columns:
        return None

    start = event_time - pd.Timedelta(minutes=window_minutes)
    end = event_time + pd.Timedelta(minutes=window_minutes)
    subset = df.loc[start:end, column].copy()

    if subset.empty:
        return None

    aligned_index = ((subset.index - event_time).total_seconds() / 60).astype(int)
    subset.index = aligned_index
    return subset


def summarize_event(aligned_df):
    return pd.DataFrame(
        {
            "median": aligned_df.median(axis=1),
            "q25": aligned_df.quantile(0.25, axis=1),
            "q75": aligned_df.quantile(0.75, axis=1),
        }
    )


def check_pretrend(summary, window=DEFAULT_PRETREND_WINDOW, tolerance=DEFAULT_PRETREND_TOL):
    pre = summary.loc[window[0] : window[1], "median"]

    if pre.empty:
        return True

    variation = pre.max() - pre.min()
    return variation <= tolerance


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


def compute_single_event_metrics(
    baseline,
    post,
    *,
    post_label="post",
    time_to_min_label="time_to_min",
    persistence_label="persistence",
):
    if baseline.empty or post.empty:
        return None

    baseline_val = baseline.median()
    post_val = post.median()
    delta = post_val - baseline_val

    if pd.isna(baseline_val) or baseline_val == 0:
        pct_change = np.nan
    else:
        pct_change = (delta / baseline_val) * 100

    below = post[post < baseline_val]
    persistence = 0 if below.empty else below.index.max() - below.index.min()
    time_to_min = np.nan if post.empty else post.idxmin()
    iqr_post = post.quantile(0.75) - post.quantile(0.25)

    return {
        "baseline": baseline_val,
        post_label: post_val,
        "delta": delta,
        "percent_change": pct_change,
        time_to_min_label: time_to_min,
        persistence_label: persistence,
        "post_iqr": iqr_post,
    }


def aggregate_event_metrics(
    metrics_list,
    *,
    post_label="post",
    time_to_min_label="time_to_min",
    persistence_label="persistence",
):
    if not metrics_list:
        return {}

    df = pd.DataFrame(metrics_list)

    return {
        "baseline": df["baseline"].median(),
        post_label: df[post_label].median(),
        "delta": df["delta"].median(),
        "percent_change": df["percent_change"].median(),
        time_to_min_label: df[time_to_min_label].median(),
        persistence_label: df[persistence_label].median(),
        "post_iqr": df["post_iqr"].median(),
        "n_events": len(df),
    }


def compute_event_metrics(
    df,
    event_col,
    signal_col,
    *,
    baseline_window=BASELINE_WINDOW,
    post_window=POST_WINDOW,
    post_label="post",
    time_to_min_label="time_to_min",
    persistence_label="persistence",
):
    if event_col not in df.columns or signal_col not in df.columns:
        return pd.DataFrame()

    on_events, off_events = detect_transitions(df, event_col)
    series = _safe_series(df, signal_col)

    if series.empty:
        return pd.DataFrame()

    results = []

    for event_type, events in {"ON": on_events, "OFF": off_events}.items():
        event_metrics = []

        for event_time in events:
            aligned = series.copy()
            rel_time = (aligned.index - event_time).total_seconds() / 60
            aligned.index = rel_time.astype(int)

            baseline = window_slice(aligned, baseline_window)
            post = window_slice(aligned, post_window)

            metrics = compute_single_event_metrics(
                baseline,
                post,
                post_label=post_label,
                time_to_min_label=time_to_min_label,
                persistence_label=persistence_label,
            )
            if metrics is not None:
                event_metrics.append(metrics)

        if event_metrics:
            agg = aggregate_event_metrics(
                event_metrics,
                post_label=post_label,
                time_to_min_label=time_to_min_label,
                persistence_label=persistence_label,
            )
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

    for col in FLOW_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill().fillna(0.0)

    df["total_gpm"] = df[FLOW_COLS].sum(axis=1)

    k = 8.34 * 1.38 * 0.66
    df["lbs_per_min"] = df["total_gpm"] * k

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

    min_periods = max(10, window // 10)
    rolling_mean = df[col].rolling(window, min_periods=min_periods).mean()
    rolling_std = df[col].rolling(window, min_periods=min_periods).std()

    return (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)


def detect_anomalies(df, col, threshold=3, window=1440):
    if col not in df.columns:
        return pd.DataFrame()

    z = add_zscore(df, col, window=window)
    anomalies = df[z.abs() >= threshold].copy()
    anomalies["z_score"] = z.loc[anomalies.index]

    return anomalies.sort_values("z_score", key=np.abs, ascending=False)
