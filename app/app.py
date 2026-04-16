import sys
from html import escape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
from scripts.constants import (
    BASELINE_WINDOW,
    DEFAULT_PRIMARY,
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
    RAW_H2S,
    RAW_NH3,
    TEMP_H2S,
    TEMP_NH3,
    WEST_GPM,
    WINDOW_48H,
)
from scripts.plotting import (
    add_event_lines_plotly as shared_add_event_lines_plotly,
    correlation_heatmap as shared_correlation_heatmap,
    dual_axis_figure as shared_dual_axis_figure,
    event_study_figure as shared_event_study_figure,
    event_window_figure as shared_event_window_figure,
    multi_panel_figure as shared_multi_panel_figure,
    scatter_with_trend as shared_scatter_with_trend,
)

try:
    from scripts.paths import PROCESSED_DATA_DIR
except Exception:
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MASTER_1MIN = PROCESSED_DATA_DIR / "master_1min.parquet"
MASTER_1H = PROCESSED_DATA_DIR / "master_1h.parquet"
MASTER_DAILY = PROCESSED_DATA_DIR / "master_daily.parquet"
MONTHLY_PATH = PROCESSED_DATA_DIR / "monthly_summary.parquet"
WEEKDAY_PATH = PROCESSED_DATA_DIR / "weekday_summary.parquet"
EVENT_METRICS_PATH = PROCESSED_DATA_DIR / "event_metrics.csv"

# --------------------------------------------------
# PAGE
# --------------------------------------------------
st.set_page_config(
    page_title="Wastewater Odor Analytics Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #f3f1ea;
        --panel: rgba(255, 252, 245, 0.9);
        --panel-strong: #fffdf7;
        --ink: #18231f;
        --muted: #5d6b66;
        --line: rgba(24, 35, 31, 0.11);
        --accent: #1f6a53;
        --accent-soft: rgba(31, 106, 83, 0.1);
        --warn: #8b5e1a;
        --shadow: 0 18px 42px rgba(31, 38, 35, 0.08);
        --radius: 18px;
    }

    html, body, [class*="css"]  {
        font-family: "IBM Plex Sans", sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(31, 106, 83, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(139, 94, 26, 0.08), transparent 24%),
            linear-gradient(180deg, #f8f5ee 0%, var(--bg) 100%);
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(18, 43, 35, 0.98) 0%, rgba(32, 64, 52, 0.96) 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #eef6f1;
    }

    [data-testid="stSidebar"] .stCaption {
        color: rgba(238, 246, 241, 0.78);
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        background: rgba(255,255,255,0.04);
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        box-shadow: var(--shadow);
    }

    [data-testid="stDataFrame"], [data-testid="stPlotlyChart"], [data-testid="stExpander"] {
        border-radius: var(--radius);
    }

    div[data-testid="stMarkdownContainer"] p code {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.92em;
        background: rgba(24, 35, 31, 0.06);
        padding: 0.12rem 0.34rem;
        border-radius: 6px;
    }

    .app-shell {
        padding-bottom: 0.75rem;
    }

    .app-hero {
        background:
            linear-gradient(135deg, rgba(255,253,247,0.96) 0%, rgba(245, 247, 242, 0.94) 100%);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 1.35rem 1.4rem 1.2rem 1.4rem;
        box-shadow: var(--shadow);
        margin: 0.15rem 0 1rem 0;
    }

    .app-kicker {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--accent);
        margin-bottom: 0.35rem;
    }

    .app-title {
        font-size: 2.2rem;
        line-height: 1.02;
        font-weight: 700;
        margin: 0;
        color: var(--ink);
    }

    .app-subtitle {
        margin-top: 0.55rem;
        max-width: 70ch;
        font-size: 1rem;
        line-height: 1.55;
        color: var(--muted);
    }

    .page-note {
        background: linear-gradient(180deg, rgba(255, 251, 241, 0.98) 0%, rgba(253, 248, 238, 0.96) 100%);
        border: 1px solid rgba(139, 94, 26, 0.18);
        border-left: 6px solid var(--warn);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin: 0.5rem 0 1rem 0;
        box-shadow: var(--shadow);
    }

    .page-note-title {
        font-size: 0.86rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--warn);
        margin-bottom: 0.45rem;
    }

    .page-note p {
        color: #473923;
        line-height: 1.58;
        margin: 0.35rem 0;
    }

    .status-pill {
        display: inline-block;
        padding: 0.28rem 0.6rem;
        margin: 0.1rem 0.45rem 0.3rem 0;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(31, 106, 83, 0.14);
    }

    .context-band {
        display: grid;
        grid-template-columns: 1.3fr 1fr;
        gap: 1rem;
        margin: 0.2rem 0 1rem 0;
    }

    .context-panel, .section-intro {
        background: linear-gradient(180deg, rgba(255, 252, 245, 0.95) 0%, rgba(250, 247, 240, 0.98) 100%);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow);
    }

    .context-panel.executive {
        background:
            linear-gradient(135deg, rgba(21, 52, 43, 0.98) 0%, rgba(33, 74, 60, 0.96) 100%);
        border-color: rgba(255,255,255,0.08);
        color: #f3f7f4;
    }

    .context-panel.executive .context-label,
    .context-panel.executive .context-title,
    .context-panel.executive .context-copy {
        color: #f3f7f4;
    }

    .context-panel.executive .context-label {
        color: rgba(243, 247, 244, 0.72);
    }

    .context-label, .section-label {
        text-transform: uppercase;
        letter-spacing: 0.11em;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--accent);
        margin-bottom: 0.35rem;
    }

    .context-title {
        font-size: 1.2rem;
        font-weight: 700;
        line-height: 1.15;
        margin-bottom: 0.28rem;
        color: var(--ink);
    }

    .context-copy, .section-copy {
        color: var(--muted);
        line-height: 1.55;
        font-size: 0.95rem;
        margin: 0;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.8rem;
    }

    .metric-pill {
        padding: 0.56rem 0.72rem;
        border-radius: 999px;
        background: rgba(31, 106, 83, 0.08);
        border: 1px solid rgba(31, 106, 83, 0.12);
        color: var(--ink);
        font-size: 0.84rem;
        line-height: 1.2;
    }

    .metric-pill strong {
        color: var(--accent);
        font-weight: 700;
        margin-right: 0.28rem;
    }

    .section-intro {
        margin: 0.2rem 0 0.8rem 0;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        line-height: 1.15;
        color: var(--ink);
        margin-bottom: 0.25rem;
    }

    .table-caption {
        font-size: 0.84rem;
        color: var(--muted);
        margin-top: -0.15rem;
        margin-bottom: 0.45rem;
    }

    .executive-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 0.35rem 0 1rem 0;
    }

    .executive-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(248, 245, 238, 0.98) 100%);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1rem 1rem 0.95rem 1rem;
        box-shadow: var(--shadow);
    }

    .executive-card-label {
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--accent);
        margin-bottom: 0.45rem;
    }

    .executive-card-value {
        font-size: 1.8rem;
        line-height: 1;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 0.45rem;
    }

    .executive-card-note {
        font-size: 0.9rem;
        line-height: 1.45;
        color: var(--muted);
    }

    .brief-list {
        margin: 0.8rem 0 0 0;
        padding-left: 1.05rem;
        color: #f3f7f4;
    }

    .brief-list li {
        margin: 0.35rem 0;
        line-height: 1.45;
    }

    .report-banner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        background: linear-gradient(90deg, rgba(31, 106, 83, 0.1) 0%, rgba(139, 94, 26, 0.08) 100%);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        margin: 0.2rem 0 1rem 0;
        box-shadow: var(--shadow);
    }

    .report-banner strong {
        display: block;
        color: var(--ink);
        margin-bottom: 0.1rem;
    }

    .report-banner span {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.45;
    }

    .report-chip {
        white-space: nowrap;
        border-radius: 999px;
        padding: 0.45rem 0.7rem;
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(24, 35, 31, 0.1);
        color: var(--accent);
        font-size: 0.82rem;
        font-weight: 700;
    }

    .report-two-col {
        display: grid;
        grid-template-columns: 1.45fr 1fr;
        gap: 1rem;
        margin: 0.25rem 0 1rem 0;
    }

    .report-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.97) 0%, rgba(248, 245, 238, 0.98) 100%);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        box-shadow: var(--shadow);
    }

    .report-card h3 {
        margin: 0 0 0.45rem 0;
        font-size: 1.05rem;
        color: var(--ink);
    }

    .report-card p, .report-card li {
        color: var(--muted);
        line-height: 1.5;
        font-size: 0.93rem;
    }

    .report-card ul {
        margin: 0.55rem 0 0 1rem;
        padding: 0;
    }

    .block-spacer {
        height: 0.35rem;
    }

    [data-testid="stRadio"] > div {
        gap: 0.5rem;
    }

    [data-testid="stRadio"] label[data-baseweb="radio"] {
        background: rgba(255, 255, 255, 0.68);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
    }

    [data-testid="stSelectbox"] label,
    [data-testid="stMultiSelect"] label,
    [data-testid="stDateInput"] label,
    [data-testid="stSlider"] label,
    [data-testid="stCheckbox"] label {
        font-weight: 600;
        color: var(--ink);
    }

    [data-testid="stAlert"] {
        border-radius: 16px;
    }

    @media (max-width: 960px) {
        .context-band {
            grid-template-columns: 1fr;
        }

        .executive-grid {
            grid-template-columns: 1fr;
        }

        .report-two-col {
            grid-template-columns: 1fr;
        }
    }

    @media print {
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"],
        [data-testid="stToolbar"],
        [data-testid="stHeader"],
        [data-testid="stRadio"],
        [data-testid="stSelectbox"],
        [data-testid="stMultiSelect"],
        [data-testid="stSlider"],
        [data-testid="stDateInput"],
        [data-testid="stCheckbox"],
        [data-testid="stDownloadButton"],
        button,
        .stTabs {
            display: none !important;
        }

        .stApp, .main, section.main > div {
            background: white !important;
        }

        .app-hero,
        .context-panel,
        .section-intro,
        .executive-card,
        .report-card,
        [data-testid="stMetric"] {
            box-shadow: none !important;
            break-inside: avoid;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOADERS
# --------------------------------------------------
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
def load_all_frames():
    master = safe_read_parquet(MASTER_1MIN)
    hourly = safe_read_parquet(MASTER_1H)
    daily = safe_read_parquet(MASTER_DAILY)
    monthly = safe_read_parquet(MONTHLY_PATH)
    weekday = safe_read_parquet(WEEKDAY_PATH)
    event_metrics = safe_read_csv(EVENT_METRICS_PATH)

    if master is not None and not isinstance(master.index, pd.DatetimeIndex):
        raise TypeError("master_1min.parquet must have a DatetimeIndex")

    if master is not None:
        master = enrich_operational_features(master)

    if hourly is None and master is not None:
        hourly = build_hourly_table(master)

    if daily is not None and isinstance(daily.index, pd.DatetimeIndex):
        daily = daily.sort_index()

    return master, hourly, daily, monthly, weekday, event_metrics


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
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


def metric_value(df, col, fn="mean", fmt="{:.2f}"):
    if not has_data(df, col):
        return "NA"
    series = df[col].dropna()
    value = getattr(series, fn)() if isinstance(fn, str) else fn(series)
    return fmt.format(value)


def coverage_value(df, col, expected_points):
    if not has_data(df, col) or expected_points <= 0:
        return "NA"
    pct = (df[col].notna().sum() / expected_points) * 100
    return f"{pct:.1f}%"


def safe_delta(series):
    if series is None or len(series.dropna()) < 2:
        return None
    clean = series.dropna()
    return clean.iloc[-1] - clean.iloc[0]


def executive_summary(master_df, events_table):
    if master_df is None or master_df.empty:
        return []

    nh3_delta = safe_delta(master_df[NH3]) if NH3 in master_df.columns else None
    h2s_delta = safe_delta(master_df[H2S]) if H2S in master_df.columns else None
    avg_flow = metric_value(master_df, "total_gpm")
    transitions = len(events_table)
    nh3_cov = coverage_value(master_df, NH3, len(master_df))
    h2s_cov = coverage_value(master_df, H2S, len(master_df))

    messages = [
        f"The reporting window covers {master_df.index.min().date()} to {master_df.index.max().date()} with {transitions:,} detected chemical transitions.",
        f"Observed gas coverage is {nh3_cov} for NH3 and {h2s_cov} for H2S, which sets the confidence ceiling for every summary shown below.",
        f"Average plant flow during the window was {avg_flow} total GPM."
    ]

    if nh3_delta is not None:
        direction = "up" if nh3_delta > 0 else "down" if nh3_delta < 0 else "flat"
        messages.append(f"NH3 finished the period {direction} by {abs(nh3_delta):.2f} ppm versus the opening level.")
    if h2s_delta is not None:
        direction = "up" if h2s_delta > 0 else "down" if h2s_delta < 0 else "flat"
        messages.append(f"H2S finished the period {direction} by {abs(h2s_delta):.2f} ppm versus the opening level.")

    return messages


def render_executive_cards(cards):
    if not cards:
        return

    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="executive-card">
                    <div class="executive-card-label">{escape(card["label"])}</div>
                    <div class="executive-card-value">{escape(card["value"])}</div>
                    <div class="executive-card-note">{escape(card["note"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_page_header(title, subtitle, kicker="Wastewater Treatment"):
    st.markdown(
        f"""
        <div class="app-shell">
            <div class="app-hero">
                <div class="app-kicker">{escape(kicker)}</div>
                <h1 class="app-title">{escape(title)}</h1>
                <div class="app-subtitle">{escape(subtitle)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_notes(page_name):
    notes = {
        "Executive Brief": """
        This page is the orientation layer for the dashboard.

        Use it to answer four questions first:
        1. What time window am I looking at?
        2. How much NH3 and H2S data is actually present in that window?
        3. Which chemical transitions occurred inside the window?
        4. Do the odor and operating signals appear to move together at a high level?

        Interpretation notes:
        - `NH3 mean` is based on the smoothed NH3 signal (`nh3_roll_mean_15min`), not the raw sensor column.
        - `H2S mean/max logic` is shown from the rolling-max-derived H2S signal because this project treats H2S as spike-driven rather than smoothly varying.
        - Coverage metrics matter. A low NH3 or H2S coverage percentage means the averages and event summaries may reflect only part of plant behavior in the selected period.
        - Event counts here are counts of transition timestamps inside the filtered window, not counts across the full project history.
        """,
        "Operations Review": """
        This page is for temporal pattern reading.

        Use the resolution selector based on the question:
        - `1-minute`: best for short spikes, event timing, and sensor behavior.
        - `1-hour`: best for relating odor to broader operating changes and hourly load context.
        - `Daily`: best for trend compression and longer-duration shifts.

        Reading guidance:
        - Dashed vertical lines are detected Ferric/HCl ON/OFF transitions.
        - Dotted purple lines are plant-level contextual events, such as the ferric reduction date.
        - When you overlay two signals, look for whether peaks, drops, or regime changes line up in time, not just whether the lines share a similar shape.
        - If a signal appears sparse or flat, check the selected window and data coverage before drawing a conclusion.
        """,
        "Event Windows": """
        This page zooms into one specific transition at a time using a ±48 hour window.

        Purpose:
        - Inspect what happened immediately before and after a chosen ON or OFF event.
        - Compare odor signals with temperature or load around the exact operational change.

        How to read it:
        - The vertical line at `0` marks the event timestamp.
        - Negative minutes are pre-event conditions; positive minutes are post-event conditions.
        - `NH3 vs Load` and `H2S vs Load` use transferred volatile load as the secondary series to help assess whether apparent odor changes may be partly explained by throughput rather than chemistry alone.

        Important caution:
        - A visible change near the event does not automatically mean the event caused it. This page is for close inspection, not causal proof by itself.
        """,
        "Event Study": """
        This page aggregates multiple aligned events of the same type.

        What is happening:
        - Each event window is aligned so the transition occurs at minute `0`.
        - The black line is the median response across aligned events.
        - The shaded band is the interquartile range (25th to 75th percentile), which shows event-to-event spread.

        Why this matters:
        - Single events can be noisy.
        - This view helps you see whether a transition tends to produce a consistent directional response across repeated occurrences.

        Pretrend note:
        - `Pretrend stable` checks whether the median signal was relatively steady before the event.
        - If pretrend stability is poor, interpretation should be more cautious because the signal may already have been changing before the transition occurred.
        """,
        "Transition Comparison": """
        This page places the first Ferric OFF, Ferric ON, HCl OFF, and HCl ON windows side by side.

        Use it when you want a visual comparison of transition archetypes rather than a single event.

        How to use it well:
        - Look for whether one transition type produces a sharper or more persistent response than another.
        - Compare both magnitude and shape. A short spike and a long suppression can have different operational meaning even if their averages are similar.
        - The metrics table below helps anchor the visual impression with summary values from the currently filtered dataset.

        Limitation:
        - This page uses representative transition windows, not every event stacked together. For multi-event behavior, use `Event Study`.
        """,
        "Performance & Coverage": """
        This page compresses the filtered window into daily, monthly, weekday, and coverage views.

        What each tab is for:
        - `Daily`: inspect day-to-day movement in odor and operating load.
        - `Monthly`: compare broad monthly operating levels within the filtered selection.
        - `Weekday`: check whether weekday structure appears in the selected period.
        - `Coverage`: verify whether missingness could distort interpretation.

        Interpretation note:
        - These aggregates are recalculated from the filtered daily dataset, so changing the sidebar date window changes the summaries shown here.
        - If the filtered period is short, monthly and weekday summaries may be descriptive but not representative.
        """,
        "Correlation & Load Analysis": """
        This page is for relationship screening, not final inference.

        What to look for:
        - The heatmap shows linear correlation structure among selected variables.
        - The scatter plot helps assess direction, spread, clustering, and whether a straight-line trend is even a reasonable summary.
        - The normalized/load-aware table is useful when you want to compare odor intensity after accounting for throughput.

        Cautions:
        - Correlation does not establish causation.
        - Relationships can change by operating regime, time scale, or data coverage.
        - A strong correlation at the daily level may not hold at the minute level, and vice versa.
        """,
        "Diagnostics & Data": """
        This page highlights unusual points using a rolling z-score.

        How it works:
        - The app compares the current value to a rolling mean and rolling standard deviation over the selected window length.
        - Large absolute z-scores indicate points that are unusual relative to recent history.

        Practical guidance:
        - Use shorter rolling windows to detect local spikes.
        - Use longer rolling windows to find larger regime departures.
        - Treat anomaly flags as candidates for review, not confirmed bad data or confirmed process upsets.

        The second chart shows the z-score itself so you can see whether anomalies are isolated bursts or part of a sustained excursion.
        """,
        "Data Explorer": """
        This page is the raw inspection surface for the filtered datasets.

        Suggested workflow:
        - Choose the dataset that matches your question.
        - Limit columns to the signals you actually want to inspect.
        - Sort by a numeric field when you want the highest values or most extreme rows first.
        - Use the CSV export when you want a quick external review of the current slice.

        This page is especially useful for validating what you saw in a chart before making any interpretation.
        """,
    }

    if page_name in notes:
        paragraphs = "".join(f"<p>{line}</p>" for line in notes[page_name].strip().split("\n\n"))
        st.markdown(
            f"""
            <div class="page-note">
                <div class="page-note-title">Executive Reading Guide</div>
                {paragraphs}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_context_band(start_ts, end_ts, rows, transitions, nh3_cov, h2s_cov):
    st.markdown(
        f"""
        <div class="context-band">
            <div class="context-panel">
                <div class="context-label">Study Window</div>
                <div class="context-title">{start_ts.date()} to {end_ts.date()}</div>
                <p class="context-copy">
                    Every chart, event count, aggregate, anomaly flag, and table on this page is scoped to the currently selected filter window.
                </p>
                <div class="pill-row">
                    <div class="metric-pill"><strong>Rows</strong>{rows:,}</div>
                    <div class="metric-pill"><strong>Transitions</strong>{transitions:,}</div>
                    <div class="metric-pill"><strong>NH3 coverage</strong>{nh3_cov}</div>
                    <div class="metric-pill"><strong>H2S coverage</strong>{h2s_cov}</div>
                </div>
            </div>
            <div class="context-panel">
                <div class="context-label">Reading Focus</div>
                <div class="context-title">Interpret the window before interpreting the signal.</div>
                <p class="context-copy">
                    Coverage, event density, and time scale all change what a pattern means. Use the filter window as part of the analysis, not just as a convenience control.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_executive_brief(master_df, events_table):
    bullets = "".join(f"<li>{escape(item)}</li>" for item in executive_summary(master_df, events_table))
    st.markdown(
        f"""
        <div class="context-panel executive">
            <div class="context-label">Executive Brief</div>
            <div class="context-title">What matters in this reporting window</div>
            <p class="context-copy">
                This summary is designed for briefing and review. It compresses the current window into a short narrative before the detailed charts.
            </p>
            <ul class="brief-list">{bullets}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_report_banner():
    st.markdown(
        """
        <div class="report-banner">
            <div>
                <strong>Printable Report Layout</strong>
                <span>
                    Use the browser print dialog on this page for a clean reporting export. Controls and navigation are suppressed in print view.
                </span>
            </div>
            <div class="report-chip">Optimized For PDF Export</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_report_highlights(master_df, events_table, event_metrics_df):
    effect_lines = []
    if event_metrics_df is not None and not event_metrics_df.empty:
        ranked = event_metrics_df.copy()
        if "percent_change" in ranked.columns:
            ranked = ranked.assign(abs_change=ranked["percent_change"].abs()).sort_values("abs_change", ascending=False)
        ranked = ranked.head(3)
        for _, row in ranked.iterrows():
            if {"chemical", "event_type", "signal", "percent_change"}.issubset(ranked.columns):
                effect_lines.append(
                    f"{row['chemical']} {row['event_type']} on {row['signal']}: {row['percent_change']:.1f}% median change."
                )

    if not effect_lines:
        effect_lines = [
            "Event effect metrics are not available for this reporting window.",
            "Use the transition tables below when a narrative explanation is still required.",
        ]

    summary_lines = executive_summary(master_df, events_table)[:3]
    summary_markup = "".join(f"<li>{escape(item)}</li>" for item in summary_lines)
    effect_markup = "".join(f"<li>{escape(item)}</li>" for item in effect_lines)

    st.markdown(
        f"""
        <div class="report-two-col">
            <div class="report-card">
                <h3>Management Summary</h3>
                <ul>{summary_markup}</ul>
            </div>
            <div class="report-card">
                <h3>Largest Estimated Effects</h3>
                <ul>{effect_markup}</ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(title, description):
    st.markdown(
        f"""
        <div class="section-intro">
            <div class="section-label">Section</div>
            <div class="section-title">{escape(title)}</div>
            <p class="section-copy">{escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dual_axis_figure(df, y1_col, y2_col, y1_label, y2_label, title, add_events=None, bar_second=False):
    return shared_dual_axis_figure(
        df,
        y1_col,
        y2_col,
        y1_label,
        y2_label,
        title,
        add_events=add_events,
        plant_events=PLANT_EVENTS if add_events else None,
        bar_second=bar_second,
    )


def event_window_figure(window_df, y1, y2, y1_label, y2_label, title, bar=False):
    return shared_event_window_figure(window_df, y1, y2, y1_label, y2_label, title, bar=bar)


def event_study_figure(summary, title, ylabel):
    return shared_event_study_figure(summary, title, ylabel)


def correlation_heatmap(df, cols):
    return shared_correlation_heatmap(df, cols)


def scatter_with_trend(df, x_col, y_col, color_col=None, title=""):
    return shared_scatter_with_trend(df, x_col, y_col, color_col=color_col, title=title)


# --------------------------------------------------
# DATA
# --------------------------------------------------
master_df, hourly_df, daily_df, monthly_df, weekday_df, event_metrics_df = load_all_frames()

if master_df is None:
    st.error(
        f"Missing required file: {MASTER_1MIN}. Run your pipeline first so the dashboard has a master 1-minute dataset to load."
    )
    st.stop()

all_events = detect_all_transitions(master_df)
events_table = build_events_table(master_df)
if event_metrics_df is None or event_metrics_df.empty:
    event_metrics_df = compute_event_metrics_table(master_df)

full_all_events = all_events
full_master_df = master_df
full_hourly_df = hourly_df
full_daily_df = daily_df
full_monthly_df = monthly_df
full_weekday_df = weekday_df
full_events_table = events_table
full_event_metrics_df = event_metrics_df


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding:0.2rem 0 0.8rem 0;">
        <div style="font-size:0.72rem; letter-spacing:0.14em; text-transform:uppercase; font-weight:700; color:rgba(238,246,241,0.72);">
            Executive Reporting
        </div>
        <div style="font-size:1.55rem; font-weight:700; line-height:1.05; margin-top:0.2rem;">
            Wastewater Odor Analytics
        </div>
        <div style="font-size:0.92rem; color:rgba(238,246,241,0.78); margin-top:0.45rem; line-height:1.45;">
            Review plant odor performance, operating transitions, process load, and data confidence inside a single reporting window.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Page",
    [
        "Executive Brief",
        "Operations Review",
        "Performance & Coverage",
        "Diagnostics & Data",
    ],
)

full_start = full_master_df.index.min().normalize()
full_end = full_master_df.index.max().normalize()

with st.sidebar.expander("Time Window", expanded=True):
    st.caption(
        "The selected dates filter the entire dashboard. "
        "Timeline, event counts, event-study summaries, aggregates, anomalies, and explorer exports all update to this window."
    )
    window_mode = st.selectbox(
        "Range",
        ["Full record", "Last 30 days", "Last 60 days", "Custom"],
        index=0,
    )

    if window_mode == "Last 30 days":
        default_start = max(full_start, full_end - pd.Timedelta(days=29))
        default_end = full_end
    elif window_mode == "Last 60 days":
        default_start = max(full_start, full_end - pd.Timedelta(days=59))
        default_end = full_end
    else:
        default_start = full_start
        default_end = full_end

    selected_range = st.date_input(
        "Dates",
        value=(default_start.date(), default_end.date()),
        min_value=full_start.date(),
        max_value=full_end.date(),
    )

    if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
        start_date, end_date = selected_range
    else:
        start_date = selected_range
        end_date = selected_range

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

master_df = filter_time_indexed_df(full_master_df, start_ts, end_ts)
hourly_df = filter_time_indexed_df(full_hourly_df, start_ts, end_ts)
daily_df = filter_time_indexed_df(full_daily_df, start_ts, end_ts)
monthly_df, weekday_df = build_period_summaries(daily_df) if daily_df is not None else (full_monthly_df, full_weekday_df)
events_table = full_events_table[
    (full_events_table["timestamp"] >= start_ts) & (full_events_table["timestamp"] <= end_ts)
].reset_index(drop=True)
all_events = {
    name: [ts for ts in times if start_ts <= ts <= end_ts]
    for name, times in full_all_events.items()
}
event_metrics_df = compute_event_metrics_table(master_df) if master_df is not None else full_event_metrics_df

if master_df is None or master_df.empty:
    st.error("The selected time window returned no rows. Adjust the sidebar date range.")
    st.stop()

with st.sidebar.expander("Data status", expanded=False):
    st.write(f"1-min: {'✅' if master_df is not None else '❌'}")
    st.write(f"1-hour: {'✅' if hourly_df is not None else '❌'}")
    st.write(f"Daily: {'✅' if daily_df is not None else '❌'}")
    st.write(f"Monthly: {'✅' if monthly_df is not None else '❌'}")
    st.write(f"Weekday: {'✅' if weekday_df is not None else '❌'}")
    st.caption(f"Filtered window: {start_ts.date()} to {end_ts.date()}")
    st.caption(
        "A green check only means the dataset is available after filtering. "
        "It does not mean the selected window has strong gas-sensor coverage."
    )

with st.sidebar.expander("Quick stats", expanded=False):
    st.write(f"Minute rows: {len(master_df):,}")
    st.write(f"NH3 coverage: {coverage_value(master_df, NH3, len(master_df))}")
    st.write(f"H2S coverage: {coverage_value(master_df, H2S, len(master_df))}")
    if daily_df is not None and not daily_df.empty:
        st.write(f"Daily rows: {len(daily_df):,}")


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
if page == "Executive Brief":
    render_page_header(
        "Wastewater Odor Performance Brief",
        "Executive summary of odor conditions, process context, transition activity, and data confidence for the currently selected reporting window.",
    )
    render_page_notes("Executive Brief")
    render_context_band(
        start_ts,
        end_ts,
        len(master_df),
        len(events_table),
        coverage_value(master_df, NH3, len(master_df)),
        coverage_value(master_df, H2S, len(master_df)),
    )
    render_report_banner()
    render_executive_brief(master_df, events_table)
    render_report_highlights(master_df, events_table, event_metrics_df)

    render_section_intro(
        "Key Performance Snapshot",
        "Use these cards as the boardroom summary: current window size, odor levels, operating load, transition activity, and confidence in the underlying sensor coverage.",
    )
    render_executive_cards(
        [
            {
                "label": "Reporting Window",
                "value": f"{master_df.index.min().date()} to {master_df.index.max().date()}",
                "note": f"{len(master_df):,} minute-level rows included in this brief.",
            },
            {
                "label": "Average Odor Load",
                "value": f"NH3 {metric_value(master_df, NH3)} | H2S {metric_value(master_df, H2S)}",
                "note": "Mean window-level odor indicators used for high-level performance review.",
            },
            {
                "label": "Process Throughput",
                "value": f"{metric_value(master_df, 'total_gpm')} GPM",
                "note": f"Average volatile transfer rate: {metric_value(master_df, 'lbs_per_min')} lbs/min.",
            },
            {
                "label": "Transition Activity",
                "value": f"{len(events_table):,}",
                "note": "Detected Ferric and HCl ON/OFF transitions inside the reporting window.",
            },
        ]
    )
    render_executive_cards(
        [
            {
                "label": "Ferric Activity",
                "value": f"ON {len(all_events.get('Ferric_ON', []))} | OFF {len(all_events.get('Ferric_OFF', []))}",
                "note": "Counts help explain whether odor movement coincides with Ferric changes.",
            },
            {
                "label": "HCl Activity",
                "value": f"ON {len(all_events.get('HCl_ON', []))} | OFF {len(all_events.get('HCl_OFF', []))}",
                "note": "Transition counts summarize HCl operational change frequency.",
            },
            {
                "label": "NH3 Data Confidence",
                "value": coverage_value(master_df, NH3, len(master_df)),
                "note": "Coverage is the share of minute rows with NH3 observations in the window.",
            },
            {
                "label": "H2S Data Confidence",
                "value": coverage_value(master_df, H2S, len(master_df)),
                "note": "Coverage is the share of minute rows with H2S observations in the window.",
            },
        ]
    )

    top_cols = available_columns(master_df, [NH3, H2S, TEMP_NH3, TEMP_H2S, "total_gpm", "lbs_per_min"])
    if top_cols:
        render_section_intro(
            "Headline Trend View",
            "Use one chart to anchor the briefing. This should show the main odor signal against the most relevant operational context for the reporting conversation.",
        )
        st.caption(
            "Select the pairing you want to use as the lead visual in the reporting narrative."
        )
        primary_left = st.selectbox("Primary signal", top_cols, index=0)
        primary_right = st.selectbox("Secondary signal", top_cols, index=min(1, len(top_cols)-1))
        fig = dual_axis_figure(
            master_df,
            primary_left,
            primary_right,
            primary_left,
            primary_right,
            f"{primary_left} vs {primary_right}",
            add_events=all_events,
        )
        st.plotly_chart(fig, use_container_width=True)

    render_section_intro(
        "Operational Transition Summary",
        "Use this section to support the narrative with concrete timing and effect estimates for operational changes observed during the reporting window.",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Detected events")
        st.markdown('<div class="table-caption">Chronological transition timestamps inside the active filter window.</div>', unsafe_allow_html=True)
        st.dataframe(events_table, use_container_width=True, height=260)
    with c2:
        st.subheader("Event response summary")
        st.markdown('<div class="table-caption">Aggregated pre/post effect metrics computed from matching transitions in the active window.</div>', unsafe_allow_html=True)
        show_cols = [
            c for c in ["chemical", "event_type", "signal", "delta", "percent_change", "time_to_min", "persistence", "post_iqr", "n_events"]
            if c in event_metrics_df.columns
        ]
        st.dataframe(event_metrics_df[show_cols] if show_cols else event_metrics_df, use_container_width=True, height=260)


# --------------------------------------------------
# FULL TIMELINE
# --------------------------------------------------
elif page == "Operations Review":
    render_page_header(
        "Full Timeline",
        f"Interactive timeline for the filtered window from {start_ts.date()} to {end_ts.date()}, with minute, hourly, and daily views.",
    )
    render_page_notes("Operations Review")
    render_context_band(
        start_ts,
        end_ts,
        len(master_df),
        len(events_table),
        coverage_value(master_df, NH3, len(master_df)),
        coverage_value(master_df, H2S, len(master_df)),
    )

    render_section_intro(
        "Interactive Timeline Builder",
        "Adjust resolution and signal pairing to match the scale of the question. Short spikes, hourly context, and longer trends should not be read from the same view.",
    )
    resolution = st.radio("Resolution", ["1-minute", "1-hour", "Daily"], horizontal=True)
    if resolution == "1-minute":
        active_df = master_df
    elif resolution == "1-hour":
        active_df = hourly_df if hourly_df is not None else master_df
    else:
        active_df = daily_df if daily_df is not None else master_df

    cols = [c for c in active_df.columns if pd.api.types.is_numeric_dtype(active_df[c])]
    default_left = NH3 if NH3 in cols else cols[0]
    default_right = H2S if H2S in cols else cols[min(1, len(cols)-1)]

    left_col = st.selectbox("Primary y-axis", cols, index=cols.index(default_left) if default_left in cols else 0)
    right_col = st.selectbox("Secondary y-axis", cols, index=cols.index(default_right) if default_right in cols else 0)
    show_events = st.checkbox("Overlay transition markers", value=True)
    st.caption(
        "Use the overlay when you are testing whether odor changes line up with Ferric or HCl transitions. "
        "Turn it off when the chart becomes too visually dense."
    )

    fig = dual_axis_figure(
        active_df,
        left_col,
        right_col,
        left_col,
        right_col,
        f"{left_col} vs {right_col} ({resolution})",
        add_events=all_events if show_events else None,
        bar_second=(resolution == "1-hour" and right_col in ["flow_gal_hr", "lbs_volatile", "fecl3_lbs"]),
    )
    st.plotly_chart(fig, use_container_width=True)

    render_section_intro(
        "Script-Aligned Shortcuts",
        "These views mirror common analysis pairings used elsewhere in the project so you can quickly reproduce familiar reads without rebuilding the chart setup each time.",
    )
    st.subheader("Script-aligned shortcuts")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(dual_axis_figure(master_df, NH3, H2S, "NH₃ (ppm)", "H₂S (ppm)", "NH₃ and H₂S – Full Timeline", add_events=all_events), use_container_width=True)
        if hourly_df is not None and has_data(hourly_df, NH3) and has_data(hourly_df, H2S):
            st.plotly_chart(dual_axis_figure(hourly_df, NH3, H2S, "NH₃ (ppm) — hourly avg", "H₂S (ppm) — hourly max", "NH₃ and H₂S – Hourly", add_events=all_events), use_container_width=True)
    with c2:
        if has_data(master_df, H2S) and has_data(master_df, TEMP_H2S):
            st.plotly_chart(dual_axis_figure(master_df, H2S, TEMP_H2S, "H₂S (ppm)", "Temperature (°F)", "H₂S and Temperature – Full Timeline", add_events=all_events), use_container_width=True)
        if hourly_df is not None and "lbs_volatile" in hourly_df.columns:
            hourly_plot = hourly_df[["lbs_volatile"]].rename(columns={"lbs_volatile": "Transferred Lbs Vol"})
            merged = master_df[[H2S]].join(hourly_plot, how="outer")
            st.plotly_chart(dual_axis_figure(merged, H2S, "Transferred Lbs Vol", "H₂S (ppm)", "Transferred Lbs Vol", "H₂S & Hourly Transferred Lbs Vol", add_events=all_events, bar_second=True), use_container_width=True)

    render_section_intro(
        "Single Transition Inspection",
        "Use this section when leadership needs a concrete before-and-after view around one specific Ferric or HCl event rather than a full-window trend.",
    )
    event_family = st.selectbox("Event family", list(EVENT_COLUMNS.keys()), key="ops_event_family")
    event_direction = st.radio("Transition", ["ON", "OFF"], horizontal=True, key="ops_event_direction")
    signal_mode = st.radio(
        "Window view",
        ["NH3 vs H2S", "NH3 vs Temperature", "H2S vs Temperature", "NH3 vs Load", "H2S vs Load"],
        horizontal=True,
        key="ops_signal_mode",
    )
    on_events, off_events = detect_transitions(master_df, EVENT_COLUMNS[event_family])
    event_times = on_events if event_direction == "ON" else off_events
    if len(event_times) == 0:
        st.warning("No events found for this selection.")
    else:
        st.metric("Available events", len(event_times))
        event_time = st.selectbox("Event timestamp", list(event_times), format_func=lambda x: x.strftime("%Y-%m-%d %H:%M"), key="ops_event_time")
        window_df = master_df.loc[event_time - WINDOW_48H : event_time + WINDOW_48H].copy()
        window_df["minutes_from_event"] = (window_df.index - event_time).total_seconds() / 60
        pairs = {
            "NH3 vs H2S": (NH3, H2S, "NH₃ (ppm)", "H₂S (ppm)", False),
            "NH3 vs Temperature": (NH3, TEMP_NH3, "NH₃ (ppm)", "Temperature (°F)", False),
            "H2S vs Temperature": (H2S, TEMP_H2S, "H₂S (ppm)", "Temperature (°F)", False),
            "NH3 vs Load": (NH3, "transferred_lbs_vol", "NH₃ (ppm)", "Transferred Vol (lbs/min equiv)", True),
            "H2S vs Load": (H2S, "transferred_lbs_vol", "H₂S (ppm)", "Transferred Vol (lbs/min equiv)", True),
        }
        y1, y2, l1, l2, bar = pairs[signal_mode]
        st.plotly_chart(
            event_window_figure(window_df, y1, y2, l1, l2, f"{signal_mode} Around {event_family} {event_direction}", bar=bar),
            use_container_width=True,
        )

    render_section_intro(
        "Repeated Event Response",
        "This view summarizes whether similar transitions tend to produce a repeatable odor response across the reporting window.",
    )
    s1, s2, s3 = st.columns(3)
    chem = s1.selectbox("Chemical", list(EVENT_COLUMNS.keys()), key="ops_study_chem")
    event_type = s2.selectbox("Event type", ["ON", "OFF"], key="ops_study_type")
    signal_label = s3.selectbox("Signal", ["NH3", "H2S"], key="ops_study_signal")
    signal_col = NH3 if signal_label == "NH3" else H2S
    summary, aligned_df, pretrend_ok = compute_event_study_summary(master_df, chem, event_type, signal_col)
    if summary is None or summary.empty:
        st.warning("No aligned event windows were available for this selection.")
    else:
        st.plotly_chart(
            event_study_figure(summary, f"{signal_label} Response Around {chem} {event_type}", f"{signal_label} (ppm)"),
            use_container_width=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Events aligned", aligned_df.shape[1])
        m2.metric("Median at event", f"{summary.loc[0, 'median']:.2f}" if 0 in summary.index else "NA")
        m3.metric("Pretrend stable", "Yes" if pretrend_ok else "No")
        m4.metric("Window", "±72h")

    render_section_intro(
        "Cross-Transition Comparison",
        "Use side-by-side windows to compare whether Ferric and HCl changes produce similar or distinct operational signatures.",
    )
    compare_options = {
        "NH3 vs H2S": (NH3, H2S, "NH₃ (ppm)", "H₂S (ppm)"),
        "NH3 vs Temperature": (NH3, TEMP_NH3, "NH₃ (ppm)", "Temperature (°F)"),
        "H2S vs Temperature": (H2S, TEMP_H2S, "H₂S (ppm)", "Temperature (°F)"),
        "NH3 vs Sludge Flow": (NH3, FLOW, "NH₃ (ppm)", "Sludge Flow (GPM)"),
        "H2S vs Sludge Flow": (H2S, FLOW, "H₂S (ppm)", "Sludge Flow (GPM)"),
    }
    choice = st.selectbox("Comparison view", list(compare_options.keys()), key="ops_compare_choice")
    y1, y2, y1_label, y2_label = compare_options[choice]
    compare_events = {}
    for chem_name, col in EVENT_COLUMNS.items():
        on_events, off_events = detect_transitions(master_df, col)
        if len(off_events) > 0:
            compare_events[f"{chem_name} OFF"] = off_events[0]
        if len(on_events) > 0:
            compare_events[f"{chem_name} ON"] = on_events[0]
    ordered = {k: compare_events[k] for k in ["Ferric OFF", "Ferric ON", "HCl OFF", "HCl ON"] if k in compare_events}
    if len(ordered) == 0:
        st.warning("No transition windows available.")
    else:
        event_windows = {}
        for event_name, center in ordered.items():
            w = master_df.loc[center - WINDOW_48H : center + WINDOW_48H].copy()
            if not w.empty:
                w["minutes"] = (w.index - center).total_seconds() / 60
            event_windows[event_name] = w
        st.plotly_chart(
            shared_multi_panel_figure(
                master_df,
                event_windows,
                y1,
                y2,
                y1_label,
                y2_label,
                f"{choice} Across Operational Transitions",
            ),
            use_container_width=True,
        )


# --------------------------------------------------
# EVENT WINDOWS
# --------------------------------------------------
elif page == "Event Windows":
    render_page_header(
        "Event Windows",
        "Inspect individual Ferric and HCl transition windows at high temporal resolution to see what changed immediately before and after each event.",
    )
    render_page_notes("Event Windows")
    render_context_band(
        start_ts,
        end_ts,
        len(master_df),
        len(events_table),
        coverage_value(master_df, NH3, len(master_df)),
        coverage_value(master_df, H2S, len(master_df)),
    )

    render_section_intro(
        "Single-Event Inspection",
        "Use this page when you want to inspect one transition closely rather than averaging across many events. The goal is local context, not generalized effect estimation.",
    )
    event_family = st.selectbox("Event family", list(EVENT_COLUMNS.keys()))
    event_direction = st.radio("Transition", ["ON", "OFF"], horizontal=True)
    signal_mode = st.radio("Window view", ["NH3 vs H2S", "NH3 vs Temperature", "H2S vs Temperature", "NH3 vs Load", "H2S vs Load"], horizontal=True)

    on_events, off_events = detect_transitions(master_df, EVENT_COLUMNS[event_family])
    event_times = on_events if event_direction == "ON" else off_events

    if len(event_times) == 0:
        st.warning("No events found for this selection.")
    else:
        st.metric("Available events", len(event_times))
        event_time = st.selectbox("Event timestamp", list(event_times), format_func=lambda x: x.strftime("%Y-%m-%d %H:%M"))
        st.caption(
            "Choose a specific transition timestamp to inspect a single event in detail. "
            "The same event family can behave differently from one occurrence to the next."
        )
        window_df = master_df.loc[event_time - WINDOW_48H : event_time + WINDOW_48H].copy()
        window_df["minutes_from_event"] = (window_df.index - event_time).total_seconds() / 60

        pairs = {
            "NH3 vs H2S": (NH3, H2S, "NH₃ (ppm)", "H₂S (ppm)", False),
            "NH3 vs Temperature": (NH3, TEMP_NH3, "NH₃ (ppm)", "Temperature (°F)", False),
            "H2S vs Temperature": (H2S, TEMP_H2S, "H₂S (ppm)", "Temperature (°F)", False),
            "NH3 vs Load": (NH3, "transferred_lbs_vol", "NH₃ (ppm)", "Transferred Vol (lbs/min equiv)", True),
            "H2S vs Load": (H2S, "transferred_lbs_vol", "H₂S (ppm)", "Transferred Vol (lbs/min equiv)", True),
        }
        y1, y2, l1, l2, bar = pairs[signal_mode]
        fig = event_window_figure(window_df, y1, y2, l1, l2, f"{signal_mode} Around {event_family} {event_direction}", bar=bar)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Window data"):
            show_cols = available_columns(window_df, [NH3, H2S, TEMP_NH3, TEMP_H2S, FLOW, "total_gpm", "lbs_per_min", "transferred_lbs_vol", "minutes_from_event"])
            st.dataframe(window_df[show_cols], use_container_width=True, height=300)


# --------------------------------------------------
# EVENT STUDY
# --------------------------------------------------
elif page == "Event Study":
    render_page_header(
        "Event Study",
        "Aggregate aligned transition windows to estimate whether a selected event type tends to produce a consistent NH3 or H2S response.",
    )
    render_page_notes("Event Study")
    render_context_band(
        start_ts,
        end_ts,
        len(master_df),
        len(events_table),
        coverage_value(master_df, NH3, len(master_df)),
        coverage_value(master_df, H2S, len(master_df)),
    )

    render_section_intro(
        "Aligned Multi-Event Response",
        "This view stacks all matching events in the filtered window at minute zero so repeated behavior is easier to distinguish from one-off noise.",
    )
    c1, c2, c3 = st.columns(3)
    chem = c1.selectbox("Chemical", list(EVENT_COLUMNS.keys()))
    event_type = c2.selectbox("Event type", ["ON", "OFF"])
    signal_label = c3.selectbox("Signal", ["NH3", "H2S"])
    signal_col = NH3 if signal_label == "NH3" else H2S
    st.caption(
        "This view pools all matching events inside the filtered window. "
        "If only one event is available, the event-study curve is descriptive but not a true multi-event summary."
    )

    summary, aligned_df, pretrend_ok = compute_event_study_summary(master_df, chem, event_type, signal_col)

    if summary is None or summary.empty:
        st.warning("No aligned event windows were available for this selection.")
    else:
        st.plotly_chart(
            event_study_figure(summary, f"{signal_label} Response Around {chem} {event_type}", f"{signal_label} (ppm)"),
            use_container_width=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Events aligned", aligned_df.shape[1])
        c2.metric("Median at event", f"{summary.loc[0, 'median']:.2f}" if 0 in summary.index else "NA")
        c3.metric("Pretrend stable", "Yes" if pretrend_ok else "No")
        c4.metric("Window", "±72h")

        st.subheader("Aligned event matrix")
        st.dataframe(aligned_df, use_container_width=True, height=250)

        st.subheader("Summary table")
        st.dataframe(summary, use_container_width=True, height=250)

        if not event_metrics_df.empty:
            st.subheader("Matching effect metrics")
            matching = event_metrics_df[
                (event_metrics_df["chemical"] == chem) &
                (event_metrics_df["event_type"] == event_type) &
                (event_metrics_df["signal"] == signal_label)
            ]
            st.dataframe(matching, use_container_width=True, height=120)


# --------------------------------------------------
# TRANSITION COMPARISON
# --------------------------------------------------
elif page == "Transition Comparison":
    render_page_header(
        "Transition Comparison",
        "Compare representative Ferric and HCl ON/OFF windows side by side to see whether operational changes share or diverge in response pattern.",
    )
    render_page_notes("Transition Comparison")
    render_context_band(
        start_ts,
        end_ts,
        len(master_df),
        len(events_table),
        coverage_value(master_df, NH3, len(master_df)),
        coverage_value(master_df, H2S, len(master_df)),
    )

    render_section_intro(
        "Representative Transition Comparison",
        "Use the paired panels to compare shape, persistence, and timing differences across Ferric and HCl transitions without leaving the same visual frame.",
    )
    compare_options = {
        "NH3 vs H2S": (NH3, H2S, "NH₃ (ppm)", "H₂S (ppm)"),
        "NH3 vs Temperature": (NH3, TEMP_NH3, "NH₃ (ppm)", "Temperature (°F)"),
        "H2S vs Temperature": (H2S, TEMP_H2S, "H₂S (ppm)", "Temperature (°F)"),
        "NH3 vs Sludge Flow": (NH3, FLOW, "NH₃ (ppm)", "Sludge Flow (GPM)"),
        "H2S vs Sludge Flow": (H2S, FLOW, "H₂S (ppm)", "Sludge Flow (GPM)"),
    }
    choice = st.selectbox("Multi-panel view", list(compare_options.keys()))
    st.caption(
        "Use this comparison to see whether a response pattern appears specific to Ferric or HCl transitions, "
        "or whether the same signal behavior shows up across multiple operational changes."
    )
    y1, y2, y1_label, y2_label = compare_options[choice]

    events = {}
    for chem_name, col in EVENT_COLUMNS.items():
        on_events, off_events = detect_transitions(master_df, col)
        if len(off_events) > 0:
            events[f"{chem_name} OFF"] = off_events[0]
        if len(on_events) > 0:
            events[f"{chem_name} ON"] = on_events[0]
    ordered = {k: events[k] for k in ["Ferric OFF", "Ferric ON", "HCl OFF", "HCl ON"] if k in events}

    if len(ordered) == 0:
        st.warning("No transition windows available.")
    else:
        event_windows = {}
        for event_name, center in ordered.items():
            w = master_df.loc[center - WINDOW_48H : center + WINDOW_48H].copy()
            if not w.empty:
                w["minutes"] = (w.index - center).total_seconds() / 60
            event_windows[event_name] = w
        fig = shared_multi_panel_figure(
            master_df,
            event_windows,
            y1,
            y2,
            y1_label,
            y2_label,
            f"{choice} Across Operational Transitions",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Response metrics")
        filtered = event_metrics_df.copy()
        if not filtered.empty:
            desired_signal = "NH3" if "NH3" in choice else "H2S" if "H2S" in choice else None
            if desired_signal and "signal" in filtered.columns:
                filtered = filtered[filtered["signal"] == desired_signal]
        st.dataframe(filtered, use_container_width=True, height=260)


# --------------------------------------------------
# AGGREGATES & COVERAGE
# --------------------------------------------------
elif page == "Performance & Coverage":
    render_page_header(
        "Aggregates & Coverage",
        "Daily, monthly, weekday, and coverage views recalculated from the current time filter so you can understand both signal level and data completeness.",
    )
    st.caption("Aggregate views are recalculated from the currently filtered daily window.")
    render_page_notes("Performance & Coverage")
    render_context_band(
        start_ts,
        end_ts,
        len(master_df),
        len(events_table),
        coverage_value(master_df, NH3, len(master_df)),
        coverage_value(master_df, H2S, len(master_df)),
    )
    render_section_intro(
        "Compressed Views",
        "Switch here when the minute-level timeline is too granular and you need daily structure, seasonal shape, weekday patterns, or a direct read on missingness.",
    )

    tabs = st.tabs(["Daily", "Monthly", "Weekday", "Coverage"])

    with tabs[0]:
        if daily_df is None:
            st.info("Daily dataset not found.")
        else:
            cols = available_columns(daily_df, [NH3, H2S, "total_gpm", "transferred_lbs_vol_daily", "h2s_std", "nh3_std"])
            left = st.selectbox("Daily primary", cols, key="daily_left")
            right = st.selectbox("Daily secondary", cols, index=min(1, len(cols)-1), key="daily_right")
            st.plotly_chart(dual_axis_figure(daily_df, left, right, left, right, f"Daily {left} vs {right}"), use_container_width=True)
            st.dataframe(daily_df[cols + available_columns(daily_df, ["n_obs_nh3", "n_obs_h2s", "n_obs_water", "nh3_coverage", "h2s_coverage", "water_coverage"])], use_container_width=True, height=280)

    with tabs[1]:
        if monthly_df is None:
            st.info("Monthly summary not found.")
        else:
            display_monthly = monthly_df.copy()
            display_monthly.index = build_month_labels(display_monthly.index)
            st.bar_chart(display_monthly[available_columns(display_monthly, ["nh3_monthly_mean", "h2s_monthly_mean", "total_gpm_monthly_mean", "transferred_lbs_vol_monthly_mean"])])
            st.dataframe(display_monthly, use_container_width=True, height=280)

    with tabs[2]:
        if weekday_df is None:
            st.info("Weekday summary not found.")
        else:
            display_weekday = weekday_df.copy()
            if "weekday_name" in display_weekday.columns:
                display_weekday = display_weekday.set_index("weekday_name")
            st.bar_chart(display_weekday[available_columns(display_weekday, ["nh3_weekday_mean", "h2s_weekday_mean", "total_gpm_weekday_mean", "transferred_lbs_vol_weekday_mean"])])
            st.dataframe(display_weekday, use_container_width=True, height=280)

    with tabs[3]:
        if daily_df is None:
            st.info("Coverage metrics require master_daily.parquet.")
        else:
            coverage_cols = available_columns(daily_df, ["nh3_coverage", "h2s_coverage", "water_coverage"])
            if coverage_cols:
                st.line_chart(daily_df[coverage_cols])
                st.dataframe(daily_df[coverage_cols + available_columns(daily_df, ["n_obs_nh3", "n_obs_h2s", "n_obs_water"])], use_container_width=True, height=260)

    render_section_intro(
        "Relationship Screening",
        "Use the correlation matrix and scatter view to support a performance narrative with simple relationship checks between odor, throughput, and operating context.",
    )
    analysis_df = daily_df.copy() if daily_df is not None else master_df.resample("1D").mean(numeric_only=True)
    analysis_df = analysis_df.copy()
    numeric_cols = [c for c in analysis_df.columns if pd.api.types.is_numeric_dtype(analysis_df[c])]
    default_corr = [c for c in [NH3, H2S, "total_gpm", "transferred_lbs_vol_daily", "transferred_lbs_vol", "nh3_std", "h2s_std", "ferric_active_lbs_per_day"] if c in numeric_cols]
    selected = st.multiselect("Columns for heatmap", numeric_cols, default=default_corr[: min(len(default_corr), 8)], key="perf_heatmap_cols")
    if len(selected) >= 2:
        st.plotly_chart(correlation_heatmap(analysis_df, selected), use_container_width=True)
    else:
        st.info("Select at least two columns for the correlation heatmap.")

    scatter_source = hourly_df if hourly_df is not None else analysis_df
    scatter_cols = [c for c in scatter_source.columns if pd.api.types.is_numeric_dtype(scatter_source[c])]
    x_default = "lbs_volatile" if "lbs_volatile" in scatter_cols else scatter_cols[0]
    y_default = H2S if H2S in scatter_cols else scatter_cols[min(1, len(scatter_cols)-1)]
    x_col = st.selectbox("Scatter x", scatter_cols, index=scatter_cols.index(x_default) if x_default in scatter_cols else 0, key="perf_scatter_x")
    y_col = st.selectbox("Scatter y", scatter_cols, index=scatter_cols.index(y_default) if y_default in scatter_cols else 0, key="perf_scatter_y")
    color_col = st.selectbox("Color by (optional)", [None] + scatter_cols, index=0, key="perf_scatter_color")
    st.plotly_chart(scatter_with_trend(scatter_source, x_col, y_col, color_col=color_col, title=f"{y_col} vs {x_col}"), use_container_width=True)


# --------------------------------------------------
# CORRELATION & LOAD ANALYSIS
# --------------------------------------------------
elif page == "Correlation & Load Analysis":
    render_page_header(
        "Correlation & Load Analysis",
        "Screen linear relationships, inspect load-aware behavior, and test whether odor intensity appears to move with process throughput or other covariates.",
    )
    render_page_notes("Correlation & Load Analysis")
    render_context_band(
        start_ts,
        end_ts,
        len(master_df),
        len(events_table),
        coverage_value(master_df, NH3, len(master_df)),
        coverage_value(master_df, H2S, len(master_df)),
    )

    analysis_df = daily_df.copy() if daily_df is not None else master_df.resample("1D").mean(numeric_only=True)
    analysis_df = analysis_df.copy()

    render_section_intro(
        "Relationship Screening",
        "Use the heatmap to narrow candidates, then use the scatter plot to test whether the apparent relationship is directional, clustered, nonlinear, or mostly noise.",
    )
    numeric_cols = [c for c in analysis_df.columns if pd.api.types.is_numeric_dtype(analysis_df[c])]
    default_corr = [c for c in [NH3, H2S, "total_gpm", "transferred_lbs_vol_daily", "transferred_lbs_vol", "nh3_std", "h2s_std", "ferric_active_lbs_per_day"] if c in numeric_cols]
    selected = st.multiselect("Columns for heatmap", numeric_cols, default=default_corr[: min(len(default_corr), 8)])
    st.caption(
        "Keep the heatmap focused. A smaller set of variables is usually easier to interpret than a large matrix with many weak correlations."
    )
    if len(selected) >= 2:
        st.plotly_chart(correlation_heatmap(analysis_df, selected), use_container_width=True)
    else:
        st.info("Select at least two columns for the correlation heatmap.")

    scatter_source = hourly_df if hourly_df is not None else analysis_df
    scatter_cols = [c for c in scatter_source.columns if pd.api.types.is_numeric_dtype(scatter_source[c])]
    x_default = "lbs_volatile" if "lbs_volatile" in scatter_cols else scatter_cols[0]
    y_default = H2S if H2S in scatter_cols else scatter_cols[min(1, len(scatter_cols)-1)]
    x_col = st.selectbox("Scatter x", scatter_cols, index=scatter_cols.index(x_default) if x_default in scatter_cols else 0)
    y_col = st.selectbox("Scatter y", scatter_cols, index=scatter_cols.index(y_default) if y_default in scatter_cols else 0)
    color_col = st.selectbox("Color by (optional)", [None] + scatter_cols, index=0)
    st.caption(
        "Coloring by a third variable can reveal regime structure, such as different behavior at higher load or under different dosing conditions."
    )
    st.plotly_chart(scatter_with_trend(scatter_source, x_col, y_col, color_col=color_col, title=f"{y_col} vs {x_col}"), use_container_width=True)

    norm_cols = available_columns(master_df, ["nh3_per_lb", "h2s_per_lb", "lbs_per_min", "total_gpm", NH3, H2S])
    if norm_cols:
        st.subheader("Normalized / load-aware variables")
        st.dataframe(master_df[norm_cols].dropna(how="all").tail(500), use_container_width=True, height=260)


# --------------------------------------------------
# ANOMALIES
# --------------------------------------------------
elif page == "Diagnostics & Data":
    render_page_header(
        "Anomalies",
        "Use rolling z-scores to identify unusual observations in odor, temperature, flow, or load-normalized signals within the filtered study window.",
    )
    render_page_notes("Diagnostics & Data")
    render_context_band(
        start_ts,
        end_ts,
        len(master_df),
        len(events_table),
        coverage_value(master_df, NH3, len(master_df)),
        coverage_value(master_df, H2S, len(master_df)),
    )

    render_section_intro(
        "Rolling Surprise Detection",
        "This workflow highlights local departures from recent history. It is useful for investigation triage, but the flags still need process context before they mean anything operationally.",
    )
    candidates = available_columns(master_df, [NH3, H2S, RAW_NH3, RAW_H2S, TEMP_NH3, TEMP_H2S, "total_gpm", "lbs_per_min", "nh3_per_lb", "h2s_per_lb"])
    target_col = st.selectbox("Signal", candidates)
    window = st.slider("Rolling window (minutes)", min_value=60, max_value=4320, value=1440, step=60)
    threshold = st.slider("Absolute z-score threshold", min_value=2.0, max_value=6.0, value=3.0, step=0.25)
    st.caption(
        "Higher thresholds show only more extreme departures. "
        "Lower thresholds are more sensitive but will usually produce more candidate anomalies to review."
    )

    z = add_zscore(master_df, target_col, window=window)
    anomalies = detect_anomalies(master_df, target_col, threshold=threshold, window=window)
    st.metric("Anomalies found", f"{len(anomalies):,}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=master_df.index, y=master_df[target_col], mode="lines", name=target_col))
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies[target_col], mode="markers", name="Anomalies", marker=dict(size=7)))
    fig.update_layout(title=f"{target_col} anomalies", template="plotly_white", hovermode="x unified", legend=dict(orientation="h"))
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    z_df = pd.DataFrame({target_col: master_df[target_col], "z_score": z}).dropna().tail(5000)
    z_fig = go.Figure()
    z_fig.add_trace(go.Scatter(x=z_df.index, y=z_df["z_score"], mode="lines", name="z-score"))
    z_fig.add_hline(y=threshold, line_dash="dash")
    z_fig.add_hline(y=-threshold, line_dash="dash")
    z_fig.update_layout(title="Rolling z-score", template="plotly_white", hovermode="x unified")
    st.plotly_chart(z_fig, use_container_width=True)

    st.dataframe(anomalies.head(500), use_container_width=True, height=280)

    render_section_intro(
        "Filtered Data Explorer",
        "Use the table view when a chart raises a question and you need to inspect underlying rows, sort extremes, or export a supporting appendix.",
    )
    dataset_name = st.selectbox(
        "Dataset",
        ["master_1min", "master_1h", "master_daily", "monthly_summary", "weekday_summary", "event_metrics"],
        key="diag_dataset",
    )
    dataset_map = {
        "master_1min": master_df,
        "master_1h": hourly_df,
        "master_daily": daily_df,
        "monthly_summary": monthly_df,
        "weekday_summary": weekday_df,
        "event_metrics": event_metrics_df,
    }
    view_df = dataset_map[dataset_name]
    if view_df is None:
        st.info(f"{dataset_name} is not available.")
    else:
        numeric_candidates = [c for c in view_df.columns if pd.api.types.is_numeric_dtype(view_df[c])]
        show_cols = st.multiselect("Columns", list(view_df.columns), default=list(view_df.columns[: min(12, len(view_df.columns))]), key="diag_show_cols")
        if not show_cols:
            show_cols = list(view_df.columns)
        max_rows = st.slider("Rows", min_value=20, max_value=2000, value=200, step=20, key="diag_rows")
        sort_col = st.selectbox("Sort by", [None] + list(view_df.columns), index=0, key="diag_sort")
        display_df = view_df.copy()
        if sort_col is not None:
            try:
                display_df = display_df.sort_values(sort_col, ascending=False)
            except Exception:
                pass
        preview_df = display_df[show_cols].head(max_rows)
        st.download_button(
            "Download selection as CSV",
            preview_df.to_csv().encode("utf-8"),
            file_name=f"{dataset_name}_{start_ts.date()}_{end_ts.date()}.csv",
            mime="text/csv",
        )
        st.dataframe(preview_df, use_container_width=True, height=420)
        if len(numeric_candidates) >= 1:
            dist_col = st.selectbox("Numeric column", numeric_candidates, key="diag_dist_col")
            hist = go.Figure(data=[go.Histogram(x=view_df[dist_col].dropna(), nbinsx=50)])
            hist.update_layout(title=f"Distribution of {dist_col}", template="plotly_white")
            st.plotly_chart(hist, use_container_width=True)


# --------------------------------------------------
# DATA EXPLORER
# --------------------------------------------------
elif page == "Data Explorer":
    render_page_header(
        "Data Explorer",
        "Inspect the underlying filtered tables directly, sort important fields, and export a quick CSV slice for external review or documentation.",
    )
    render_page_notes("Data Explorer")
    render_context_band(
        start_ts,
        end_ts,
        len(master_df),
        len(events_table),
        coverage_value(master_df, NH3, len(master_df)),
        coverage_value(master_df, H2S, len(master_df)),
    )

    render_section_intro(
        "Table-Level Validation",
        "Use the explorer to validate values behind charts, inspect extreme rows directly, and export a filtered slice without leaving the dashboard.",
    )
    dataset_name = st.selectbox("Dataset", ["master_1min", "master_1h", "master_daily", "monthly_summary", "weekday_summary", "event_metrics"])
    st.caption(
        "The explorer always reflects the current sidebar time filter for time-indexed datasets. "
        "Use it to validate specific values behind charts before making an interpretation."
    )
    dataset_map = {
        "master_1min": master_df,
        "master_1h": hourly_df,
        "master_daily": daily_df,
        "monthly_summary": monthly_df,
        "weekday_summary": weekday_df,
        "event_metrics": event_metrics_df,
    }
    view_df = dataset_map[dataset_name]

    if view_df is None:
        st.info(f"{dataset_name} is not available.")
    else:
        if isinstance(view_df, pd.DataFrame):
            numeric_candidates = [c for c in view_df.columns if pd.api.types.is_numeric_dtype(view_df[c])]
            show_cols = st.multiselect("Columns", list(view_df.columns), default=list(view_df.columns[: min(12, len(view_df.columns))]))
            if not show_cols:
                show_cols = list(view_df.columns)
            max_rows = st.slider("Rows", min_value=20, max_value=2000, value=200, step=20)
            sort_col = st.selectbox("Sort by", [None] + list(view_df.columns), index=0)

            display_df = view_df.copy()
            if sort_col is not None:
                try:
                    display_df = display_df.sort_values(sort_col, ascending=False)
                except Exception:
                    pass
            preview_df = display_df[show_cols].head(max_rows)
            st.download_button(
                "Download selection as CSV",
                preview_df.to_csv().encode("utf-8"),
                file_name=f"{dataset_name}_{start_ts.date()}_{end_ts.date()}.csv",
                mime="text/csv",
            )
            st.dataframe(preview_df, use_container_width=True, height=500)

            if len(numeric_candidates) >= 1:
                st.subheader("Quick distribution")
                dist_col = st.selectbox("Numeric column", numeric_candidates)
                hist = go.Figure(data=[go.Histogram(x=view_df[dist_col].dropna(), nbinsx=50)])
                hist.update_layout(title=f"Distribution of {dist_col}", template="plotly_white")
                st.plotly_chart(hist, use_container_width=True)
