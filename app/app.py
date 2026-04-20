import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from scripts.constants import (
    H2S,
    NH3,
)
try:
    from scripts.paths import PROCESSED_DATA_DIR
except Exception:
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

from app.dashboard_ui import (
    APP_STYLE,
    coverage_value,
    metric_value,
)
from app.app_state import (
    build_filtered_state,
    render_page_selector,
    render_sidebar_header,
    render_sidebar_resources,
    select_time_window,
)
from app.data_services import (
    add_zscore,
    available_columns,
    build_events_table,
    build_month_labels,
    build_period_summaries,
    compute_event_metrics_table,
    compute_event_study_summary,
    correlation_heatmap,
    detect_all_transitions,
    detect_anomalies,
    detect_transitions,
    dual_axis_figure,
    event_study_figure,
    event_window_figure,
    filter_time_indexed_df,
    has_data,
    load_all_frames,
    safe_read_csv,
    scatter_with_trend,
)
from app.page_renderers import PAGE_OPTIONS, render_page


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MASTER_1MIN = PROCESSED_DATA_DIR / "master_1min.parquet"
MASTER_1H = PROCESSED_DATA_DIR / "master_1h.parquet"
MASTER_DAILY = PROCESSED_DATA_DIR / "master_daily.parquet"
MONTHLY_PATH = PROCESSED_DATA_DIR / "monthly_summary.parquet"
WEEKDAY_PATH = PROCESSED_DATA_DIR / "weekday_summary.parquet"
EVENT_METRICS_PATH = PROCESSED_DATA_DIR / "event_metrics.csv"
STRUVITE_OBS_PATH = PROCESSED_DATA_DIR / "struvite_observations.csv"
CHEM_LABS_PATH = PROCESSED_DATA_DIR / "chemistry_lab_results.csv"
METHODS_LOG_PATH = PROJECT_ROOT / "notes" / "methods_log.csv"
THESIS_STATUS_PATH = PROJECT_ROOT / "notes" / "thesis_outline_status.md"

# --------------------------------------------------
# PAGE
# --------------------------------------------------
st.set_page_config(
    page_title="Wastewater Odor Analytics Dashboard",
    layout="wide",
)

st.markdown(APP_STYLE, unsafe_allow_html=True)


# --------------------------------------------------
# DATA
# --------------------------------------------------
master_df, hourly_df, daily_df, monthly_df, weekday_df, event_metrics_df, struvite_obs_df, chem_labs_df = load_all_frames(
    MASTER_1MIN,
    MASTER_1H,
    MASTER_DAILY,
    MONTHLY_PATH,
    WEEKDAY_PATH,
    EVENT_METRICS_PATH,
    STRUVITE_OBS_PATH,
    CHEM_LABS_PATH,
)
methods_log_df = safe_read_csv(METHODS_LOG_PATH)
thesis_status_text = THESIS_STATUS_PATH.read_text() if THESIS_STATUS_PATH.exists() else ""

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

full_state = {
    "master_df": full_master_df,
    "hourly_df": full_hourly_df,
    "daily_df": full_daily_df,
    "monthly_df": full_monthly_df,
    "weekday_df": full_weekday_df,
    "events_table": full_events_table,
    "all_events": full_all_events,
    "event_metrics_df": full_event_metrics_df,
    "struvite_obs_df": struvite_obs_df,
    "chem_labs_df": chem_labs_df,
    "methods_log_df": methods_log_df,
    "thesis_status_text": thesis_status_text,
}


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
render_sidebar_header()
page = render_page_selector(PAGE_OPTIONS)
window_state = select_time_window(full_master_df)
filtered_state = build_filtered_state(
    full_state,
    window_state,
    filter_time_indexed_df,
    build_period_summaries,
    compute_event_metrics_table,
)

master_df = filtered_state["master_df"]
hourly_df = filtered_state["hourly_df"]
daily_df = filtered_state["daily_df"]
monthly_df = filtered_state["monthly_df"]
weekday_df = filtered_state["weekday_df"]
events_table = filtered_state["events_table"]
all_events = filtered_state["all_events"]
event_metrics_df = filtered_state["event_metrics_df"]
start_ts = filtered_state["start_ts"]
end_ts = filtered_state["end_ts"]
struvite_obs_df = filtered_state["struvite_obs_df"]
chem_labs_df = filtered_state["chem_labs_df"]

if master_df is None or master_df.empty:
    st.error("The selected time window returned no rows. Adjust the sidebar date range.")
    st.stop()

render_sidebar_resources(master_df, hourly_df, daily_df, monthly_df, weekday_df, start_ts, end_ts, coverage_value)

render_page(
    page,
    {
        "master_df": master_df,
        "hourly_df": hourly_df,
        "daily_df": daily_df,
        "monthly_df": monthly_df,
        "weekday_df": weekday_df,
        "event_metrics_df": event_metrics_df,
        "events_table": events_table,
        "all_events": all_events,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "struvite_obs_df": struvite_obs_df,
        "chem_labs_df": chem_labs_df,
        "methods_log_df": methods_log_df,
        "thesis_status_text": thesis_status_text,
        "has_data": has_data,
        "available_columns": available_columns,
        "detect_transitions": detect_transitions,
        "compute_event_study_summary": compute_event_study_summary,
        "add_zscore": add_zscore,
        "detect_anomalies": detect_anomalies,
        "build_month_labels": build_month_labels,
        "metric_value": metric_value,
        "coverage_value": coverage_value,
        "dual_axis_figure": dual_axis_figure,
        "event_window_figure": event_window_figure,
        "event_study_figure": event_study_figure,
        "correlation_heatmap": correlation_heatmap,
        "scatter_with_trend": scatter_with_trend,
    },
)
