from pathlib import Path

import pandas as pd
import streamlit as st

from scripts.analytics import (
    add_operational_features as shared_add_operational_features,
    add_zscore as shared_add_zscore,
    aggregate_event_metrics as shared_aggregate_event_metrics,
    check_pretrend as shared_check_pretrend,
    compute_single_event_metrics as shared_compute_single_event_metrics,
    detect_all_transitions as shared_detect_all_transitions,
    detect_anomalies as shared_detect_anomalies,
    detect_transitions as shared_detect_transitions,
    extract_event_window as shared_extract_event_window,
    summarize_event as shared_summarize_event,
    window_slice as shared_window_slice,
)
from scripts.chemistry_features import add_ferric_dose_features, add_hcl_dose_features
from scripts.constants import (
    BASELINE_WINDOW,
    DIG_GPM,
    EAST_GPM,
    EVENT_COLUMNS,
    EVENT_STUDY_WINDOW,
    FLOW,
    H2S,
    NH3,
    PLANT_EVENTS,
    POST_WINDOW,
    PRETREND_TOL,
    PRETREND_WINDOW,
    TEMP_H2S,
    TEMP_NH3,
    WEST_GPM,
)
from scripts.plotting import (
    add_event_lines_plotly as shared_add_event_lines_plotly,
    correlation_heatmap as shared_correlation_heatmap,
    dual_axis_figure as shared_dual_axis_figure,
    event_study_figure as shared_event_study_figure,
    event_window_figure as shared_event_window_figure,
    scatter_with_trend as shared_scatter_with_trend,
)


@st.cache_data(show_spinner=False)
def safe_read_parquet(path: Path):
    if path.exists():
        df = pd.read_parquet(path)
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        return df
    return None


@st.cache_data(show_spinner=False)
def safe_read_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def safe_read_csv_dates(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, parse_dates=True)
    except Exception:
        return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_all_frames(master_1min, master_1h, master_daily, monthly_path, weekday_path, event_metrics_path, struvite_obs_path, chem_labs_path):
    master = safe_read_parquet(master_1min)
    hourly = safe_read_parquet(master_1h)
    daily = safe_read_parquet(master_daily)
    monthly = safe_read_parquet(monthly_path)
    weekday = safe_read_parquet(weekday_path)
    event_metrics = safe_read_csv(event_metrics_path)

    if master is not None and not isinstance(master.index, pd.DatetimeIndex):
        raise TypeError("master_1min.parquet must have a DatetimeIndex")

    if master is not None:
        master = enrich_operational_features(master)

    if hourly is None and master is not None:
        hourly = build_hourly_table(master)

    if daily is not None and isinstance(daily.index, pd.DatetimeIndex):
        daily = daily.sort_index()

    struvite_obs = safe_read_csv_dates(struvite_obs_path)
    chem_labs = safe_read_csv_dates(chem_labs_path)

    return master, hourly, daily, monthly, weekday, event_metrics, struvite_obs, chem_labs


def has_data(df, col):
    return df is not None and col in df.columns and df[col].notna().any()


def available_columns(df, candidates):
    return [c for c in candidates if df is not None and c in df.columns]


def detect_transitions(df, column):
    if df is None:
        return pd.Index([]), pd.Index([])
    return shared_detect_transitions(df, column)


def detect_all_transitions(df, event_columns=EVENT_COLUMNS):
    if df is None:
        return {}
    return shared_detect_all_transitions(df, event_columns)


def build_events_table(df):
    records = []
    if df is None:
        return pd.DataFrame()

    for chem_name, col in EVENT_COLUMNS.items():
        on_events, off_events = detect_transitions(df, col)
        for ts in on_events:
            records.append({"timestamp": ts, "chemical": chem_name, "event_type": "ON"})
        for ts in off_events:
            records.append({"timestamp": ts, "chemical": chem_name, "event_type": "OFF"})

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def add_event_lines_plotly(fig, events, yref="paper", include_labels=True):
    shared_add_event_lines_plotly(
        fig,
        events,
        yref=yref,
        include_labels=include_labels,
        plant_events=PLANT_EVENTS,
    )


def enrich_operational_features(df):
    if df is None:
        return None
    df = shared_add_operational_features(df)
    if FLOW not in df.columns:
        df[FLOW] = df[[WEST_GPM, EAST_GPM, DIG_GPM]].sum(axis=1)
    df["transferred_lbs_vol"] = df["lbs_per_min"]

    eps = 1e-9
    if has_data(df, NH3):
        df["nh3_per_lb"] = df[NH3] / (df["lbs_per_min"] + eps)
    if has_data(df, H2S):
        df["h2s_per_lb"] = df[H2S] / (df["lbs_per_min"] + eps)

    if "ferric_available" in df.columns and "ferric_active_lbs_per_day" not in df.columns:
        df = add_ferric_dose_features(df)
    if "hcl_available" in df.columns and "hcl_active_lbs_per_day" not in df.columns:
        df = add_hcl_dose_features(df)

    return df


def build_hourly_table(df):
    if df is None:
        return None

    df = enrich_operational_features(df)
    hourly = pd.DataFrame(index=df.resample("1h", label="right", closed="right").size().index)
    hourly["flow_gal_hr"] = df["total_gpm"].resample("1h", label="right", closed="right").sum()
    hourly["lbs_volatile"] = df["lbs_per_min"].resample("1h", label="right", closed="right").sum()
    hourly["fecl3_lbs"] = hourly["lbs_volatile"] / 24.3

    agg = {}
    if has_data(df, H2S):
        agg[H2S] = "max"
    if has_data(df, NH3):
        agg[NH3] = "mean"
    if has_data(df, TEMP_NH3):
        agg[TEMP_NH3] = "mean"
    if has_data(df, TEMP_H2S):
        agg[TEMP_H2S] = "mean"
    if agg:
        hourly = hourly.join(df.resample("1h", label="right", closed="right").agg(agg))

    return hourly


def window_slice(series, window):
    return shared_window_slice(series, window)


def compute_single_event_metrics(baseline, post):
    return shared_compute_single_event_metrics(baseline, post)


def aggregate_event_metrics(metrics_list):
    return shared_aggregate_event_metrics(metrics_list)


@st.cache_data(show_spinner=False)
def compute_event_metrics_table(df):
    if df is None:
        return pd.DataFrame()

    results = []
    targets = {"NH3": NH3, "H2S": H2S}

    for chem_name, event_col in EVENT_COLUMNS.items():
        if event_col not in df.columns:
            continue
        on_events, off_events = detect_transitions(df, event_col)
        for event_type, event_times in {"ON": on_events, "OFF": off_events}.items():
            for signal_name, signal_col in targets.items():
                if signal_col not in df.columns:
                    continue
                event_metrics = []
                series = df[signal_col].dropna()
                if series.empty:
                    continue
                for t in event_times:
                    rel = (series.index - t).total_seconds() / 60
                    aligned = series.copy()
                    aligned.index = rel.astype(int)
                    baseline = window_slice(aligned, BASELINE_WINDOW)
                    post = window_slice(aligned, POST_WINDOW)
                    metrics = compute_single_event_metrics(baseline, post)
                    if metrics is not None:
                        event_metrics.append(metrics)
                if event_metrics:
                    agg = aggregate_event_metrics(event_metrics)
                    results.append(
                        {
                            "chemical": chem_name,
                            "event_type": event_type,
                            "signal": signal_name,
                            **agg,
                        }
                    )

    out = pd.DataFrame(results)
    if not out.empty:
        out = out.sort_values(["chemical", "event_type", "signal"]).reset_index(drop=True)
    return out


def extract_event_window(df, event_time, column, window_minutes=EVENT_STUDY_WINDOW):
    if df is None:
        return None
    return shared_extract_event_window(df, event_time, column, window_minutes)


def summarize_event(aligned_df):
    return shared_summarize_event(aligned_df)


def check_pretrend(summary, window=PRETREND_WINDOW, tolerance=PRETREND_TOL):
    return shared_check_pretrend(summary, window=window, tolerance=tolerance)


@st.cache_data(show_spinner=False)
def compute_event_study_summary(df, chem_name, event_type, signal_col):
    if df is None or signal_col not in df.columns:
        return None, pd.DataFrame(), True
    event_col = EVENT_COLUMNS[chem_name]
    on_events, off_events = detect_transitions(df, event_col)
    event_times = on_events if event_type == "ON" else off_events

    aligned_windows = []
    for t in event_times:
        window = extract_event_window(df, t, signal_col, EVENT_STUDY_WINDOW)
        if window is not None:
            aligned_windows.append(window)

    if not aligned_windows:
        return None, pd.DataFrame(), True

    aligned_df = pd.concat(aligned_windows, axis=1)
    aligned_df.columns = [f"event_{i+1}" for i in range(aligned_df.shape[1])]
    summary = summarize_event(aligned_df)
    stable = check_pretrend(summary)
    return summary, aligned_df, stable


def add_zscore(df, col, window=1440):
    if df is None or col not in df.columns:
        return pd.Series(dtype=float)
    return shared_add_zscore(df, col, window=window)


def detect_anomalies(df, col, threshold=3.0, window=1440):
    if df is None:
        return pd.DataFrame()
    return shared_detect_anomalies(df, col, threshold=threshold, window=window)


def build_month_labels(index_like):
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    return [month_names.get(int(x), str(x)) for x in index_like]


def numeric_columns(df):
    if df is None:
        return []
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def filter_time_indexed_df(df, start_ts, end_ts):
    if df is None or not isinstance(df.index, pd.DatetimeIndex):
        return df
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()


def build_period_summaries(daily):
    if daily is None or daily.empty or not isinstance(daily.index, pd.DatetimeIndex):
        return None, None

    daily = daily.sort_index().copy()
    daily["month"] = daily.index.month
    daily["weekday"] = daily.index.dayofweek

    numeric_cols = numeric_columns(daily)
    excluded = {"month", "weekday"}
    agg_cols = [c for c in numeric_cols if c not in excluded]

    if not agg_cols:
        return None, None

    monthly = daily.groupby("month")[agg_cols].mean().sort_index()
    weekday = daily.groupby("weekday")[agg_cols].mean().sort_index()

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
    monthly["days_in_data"] = daily.groupby("month").size()

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
    weekday["days_in_data"] = daily.groupby("weekday").size()
    weekday["weekday_name"] = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ][: len(weekday)]

    return monthly, weekday


def dual_axis_figure(
    df,
    y1_col,
    y2_col,
    y1_label,
    y2_label,
    title,
    add_events=None,
    bar_second=False,
    y1_scale_mode="auto",
    y2_scale_mode="auto",
):
    kwargs = dict(
        add_events=add_events,
        plant_events=PLANT_EVENTS if add_events else None,
        bar_second=bar_second,
    )
    try:
        return shared_dual_axis_figure(
            df,
            y1_col,
            y2_col,
            y1_label,
            y2_label,
            title,
            y1_scale_mode=y1_scale_mode,
            y2_scale_mode=y2_scale_mode,
            **kwargs,
        )
    except TypeError as exc:
        if "y1_scale_mode" not in str(exc) and "y2_scale_mode" not in str(exc):
            raise
        return shared_dual_axis_figure(
            df,
            y1_col,
            y2_col,
            y1_label,
            y2_label,
            title,
            **kwargs,
        )


def event_window_figure(window_df, y1, y2, y1_label, y2_label, title, bar=False):
    return shared_event_window_figure(window_df, y1, y2, y1_label, y2_label, title, bar=bar)


def event_study_figure(summary, title, ylabel):
    return shared_event_study_figure(summary, title, ylabel)


def correlation_heatmap(df, cols):
    return shared_correlation_heatmap(df, cols)


def scatter_with_trend(df, x_col, y_col, color_col=None, title=""):
    return shared_scatter_with_trend(df, x_col, y_col, color_col=color_col, title=title)
