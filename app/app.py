import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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

NH3 = "nh3_roll_mean_15min"
H2S = "h2s_roll_max_15min"
RAW_NH3 = "nh3_nh3_ppm"
RAW_H2S = "h2s_h2s_ppm"
TEMP_NH3 = "nh3_temperature_°f"
TEMP_H2S = "h2s_temperature_°f"
FLOW = "east_sludge_out_gpm_combined"
WEST_GPM = "west_sludge_out_gpm"
EAST_GPM = "east_sludge_out_gpm"
DIG_GPM = "digesters_sludge_out_flow"

EVENT_COLUMNS = {
    "Ferric": "ferric_available",
    "HCl": "hcl_available",
}

PLANT_EVENTS = {
    "Ferric Reduced": pd.Timestamp("2026-01-07"),
}

BASELINE_WINDOW = (-48 * 60, -12 * 60)
POST_WINDOW = (12 * 60, 96 * 60)
EVENT_STUDY_WINDOW = 72 * 60
PRETREND_WINDOW = (-1440, -60)
PRETREND_TOL = 0.1
WINDOW_48H = pd.Timedelta(hours=48)

DEFAULT_PRIMARY = [NH3, H2S, TEMP_NH3, TEMP_H2S, FLOW]


# --------------------------------------------------
# PAGE
# --------------------------------------------------
st.set_page_config(
    page_title="Wastewater Odor Analytics Dashboard",
    layout="wide",
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
    if df is None or column not in df.columns:
        return pd.Index([]), pd.Index([])
    series = df[column].fillna(0).astype(int)
    diff = series.diff()
    return df.index[diff == 1], df.index[diff == -1]


def detect_all_transitions(df, event_columns=EVENT_COLUMNS):
    events = {}
    if df is None:
        return events
    for name, col in event_columns.items():
        on_events, off_events = detect_transitions(df, col)
        events[f"{name}_ON"] = sorted(set(on_events))
        events[f"{name}_OFF"] = sorted(set(off_events))
    return events


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
    for name, times in events.items():
        for t in times:
            fig.add_vline(x=t, line_dash="dash", line_width=1, opacity=0.3)
            if include_labels:
                fig.add_annotation(
                    x=t,
                    y=1,
                    yref=yref,
                    text=name,
                    textangle=-90,
                    showarrow=False,
                    xanchor="left",
                    yanchor="top",
                    font=dict(size=8),
                )

    for name, t in PLANT_EVENTS.items():
        fig.add_vline(x=t, line_dash="dot", line_color="purple", line_width=1.5)
        if include_labels:
            fig.add_annotation(
                x=t,
                y=1,
                yref=yref,
                text=name,
                textangle=-90,
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(size=9, color="purple"),
            )


def enrich_operational_features(df):
    if df is None:
        return None
    df = df.copy()

    for col in [WEST_GPM, EAST_GPM, DIG_GPM]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill().fillna(0)

    if FLOW not in df.columns:
        df[FLOW] = df[[WEST_GPM, EAST_GPM, DIG_GPM]].sum(axis=1)

    df["total_gpm"] = df[[WEST_GPM, EAST_GPM, DIG_GPM]].sum(axis=1)
    k = 8.34 * 1.38 * 0.66
    df["lbs_per_min"] = df["total_gpm"] * k
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
    if series is None or series.empty:
        return pd.Series(dtype=float)
    return series.loc[(series.index >= window[0]) & (series.index <= window[1])]


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

    below = post[post < baseline_val]
    persistence = 0 if below.empty else below.index.max() - below.index.min()
    time_to_min = np.nan if post.empty else post.idxmin()
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
    if df is None or column not in df.columns:
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


def check_pretrend(summary, window=PRETREND_WINDOW, tolerance=PRETREND_TOL):
    pre = summary.loc[window[0] : window[1], "median"]
    if pre.empty:
        return True
    variation = pre.max() - pre.min()
    return variation <= tolerance


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
    rolling_mean = df[col].rolling(window, min_periods=max(10, window // 10)).mean()
    rolling_std = df[col].rolling(window, min_periods=max(10, window // 10)).std()
    return (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)


def detect_anomalies(df, col, threshold=3.0, window=1440):
    if df is None or col not in df.columns:
        return pd.DataFrame()
    z = add_zscore(df, col, window=window)
    out = df.loc[z.abs() >= threshold, [col]].copy()
    out["z_score"] = z.loc[out.index]
    return out.sort_values("z_score", key=np.abs, ascending=False)


def build_month_labels(index_like):
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    return [month_names.get(int(x), str(x)) for x in index_like]


def dual_axis_figure(df, y1_col, y2_col, y1_label, y2_label, title, add_events=None, bar_second=False):
    fig = go.Figure()

    if not has_data(df, y1_col):
        return fig

    plot_df = df[[c for c in [y1_col, y2_col] if c in df.columns]].dropna(how="all").copy()
    if plot_df.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df[y1_col],
            mode="lines",
            name=y1_label,
            line=dict(width=2),
            yaxis="y",
            hovertemplate="Time: %{x}<br>" + f"{y1_label}: " + "%{y:.2f}<extra></extra>",
        )
    )

    if y2_col in plot_df.columns:
        if bar_second:
            fig.add_trace(
                go.Bar(
                    x=plot_df.index,
                    y=plot_df[y2_col],
                    name=y2_label,
                    yaxis="y2",
                    hovertemplate="Time: %{x}<br>" + f"{y2_label}: " + "%{y:.2f}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df[y2_col],
                    mode="lines",
                    name=y2_label,
                    line=dict(width=1.6, dash="dot"),
                    yaxis="y2",
                    hovertemplate="Time: %{x}<br>" + f"{y2_label}: " + "%{y:.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(title="Date / Time"),
        yaxis=dict(title=y1_label),
        yaxis2=dict(title=y2_label, overlaying="y", side="right"),
        legend=dict(orientation="h"),
        barmode="overlay" if bar_second else None,
        margin=dict(l=100, r=100, t=100, b=100),
    )

    if add_events:
        add_event_lines_plotly(fig, add_events)
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def event_window_figure(window_df, y1, y2, y1_label, y2_label, title, bar=False):
    fig = go.Figure()

    if window_df.empty or y1 not in window_df.columns:
        return fig

    fig.add_trace(
        go.Scatter(
            x=window_df["minutes_from_event"],
            y=window_df[y1],
            mode="lines",
            name=y1_label,
            line=dict(width=2),
            customdata=window_df.index,
            hovertemplate="Time: %{customdata}<br>" + f"{y1_label}: " + "%{y:.2f}<br>Δmin: %{x}<extra></extra>",
        )
    )

    if y2 in window_df.columns:
        if bar:
            fig.add_trace(
                go.Bar(
                    x=window_df["minutes_from_event"],
                    y=window_df[y2],
                    name=y2_label,
                    opacity=0.6,
                    yaxis="y2",
                    hovertemplate=f"{y2_label}: " + "%{y:.2f}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=window_df["minutes_from_event"],
                    y=window_df[y2],
                    mode="lines",
                    name=y2_label,
                    line=dict(dash="dot"),
                    yaxis="y2",
                    hovertemplate=f"{y2_label}: " + "%{y:.2f}<extra></extra>",
                )
            )

    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(title="Minutes from Event"),
        yaxis=dict(title=y1_label),
        yaxis2=dict(title=y2_label, overlaying="y", side="right"),
        legend=dict(orientation="h"),
        margin=dict(l=100, r=100, t=100, b=100),
    )
    return fig


def event_study_figure(summary, title, ylabel):
    fig = go.Figure()
    if summary is None or summary.empty:
        return fig

    fig.add_trace(go.Scatter(x=summary.index, y=summary["q75"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=summary.index,
            y=summary["q25"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(70,130,180,0.3)",
            line=dict(width=0),
            name="IQR (25–75%)",
            hovertemplate="Q25: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary.index,
            y=summary["median"],
            mode="lines",
            name="Median",
            line=dict(width=3, color="black"),
            hovertemplate="Median: %{y:.2f}<br>Δmin: %{x}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Event", annotation_position="top")
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Minutes from event",
        yaxis_title=ylabel,
        legend=dict(orientation="h"),
        margin=dict(l=100, r=100, t=100, b=100),
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def correlation_heatmap(df, cols):
    corr = df[cols].corr(numeric_only=True)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Correlation Heatmap",
        template="plotly_white",
        height=650,
        margin=dict(l=100, r=100, t=100, b=100),
    )
    return fig


def scatter_with_trend(df, x_col, y_col, color_col=None, title=""):
    plot_df = df[[c for c in [x_col, y_col, color_col] if c and c in df.columns]].dropna().copy()
    fig = go.Figure()
    if plot_df.empty:
        return fig

    marker_kwargs = dict(size=6, opacity=0.55)
    if color_col and color_col in plot_df.columns:
        marker_kwargs["color"] = plot_df[color_col]
        marker_kwargs["colorscale"] = "Viridis"
        marker_kwargs["showscale"] = True
        marker_kwargs["colorbar"] = {"title": color_col}

    fig.add_trace(
        go.Scatter(
            x=plot_df[x_col],
            y=plot_df[y_col],
            mode="markers",
            marker=marker_kwargs,
            name="Observations",
            text=plot_df.index.astype(str) if isinstance(plot_df.index, pd.DatetimeIndex) else None,
            hovertemplate=f"{x_col}: %{{x:.3f}}<br>{y_col}: %{{y:.3f}}<extra></extra>",
        )
    )

    if len(plot_df) >= 2:
        x = plot_df[x_col].astype(float).to_numpy()
        y = plot_df[y_col].astype(float).to_numpy()
        if np.isfinite(x).all() and np.isfinite(y).all() and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=slope * xs + intercept,
                    mode="lines",
                    name=f"Trend (slope={slope:.3f})",
                )
            )

    fig.update_layout(
        title=title or f"{y_col} vs {x_col}",
        template="plotly_white",
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend=dict(orientation="h"),
        margin=dict(l=100, r=100, t=100, b=100),
    )
    return fig


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


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("Wastewater Treatment")
page = st.sidebar.radio(
    "Page",
    [
        "Overview",
        "Full Timeline",
        "Event Windows",
        "Event Study",
        "Transition Comparison",
        "Aggregates & Coverage",
        "Correlation & Load Analysis",
        "Anomalies",
        "Data Explorer",
    ],
)

with st.sidebar.expander("Data status", expanded=False):
    st.write(f"1-min: {'✅' if master_df is not None else '❌'}")
    st.write(f"1-hour: {'✅' if hourly_df is not None else '❌'}")
    st.write(f"Daily: {'✅' if daily_df is not None else '❌'}")
    st.write(f"Monthly: {'✅' if monthly_df is not None else '❌'}")
    st.write(f"Weekday: {'✅' if weekday_df is not None else '❌'}")


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
if page == "Overview":
    st.title("Wastewater Odor Analytics Dashboard")
    st.caption("Integrated view of odor, operations, transitions, load, and summary diagnostics.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (1-min)", f"{len(master_df):,}")
    c2.metric("NH3 mean", f"{master_df[NH3].mean():.2f}" if has_data(master_df, NH3) else "NA")
    c3.metric("H2S mean/max logic", f"{master_df[H2S].mean():.2f}" if has_data(master_df, H2S) else "NA")
    c4.metric("Date range", f"{master_df.index.min().date()} → {master_df.index.max().date()}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ferric ON events", len(all_events.get("Ferric_ON", [])))
    c2.metric("Ferric OFF events", len(all_events.get("Ferric_OFF", [])))
    c3.metric("HCl ON events", len(all_events.get("HCl_ON", [])))
    c4.metric("HCl OFF events", len(all_events.get("HCl_OFF", [])))

    top_cols = available_columns(master_df, [NH3, H2S, TEMP_NH3, TEMP_H2S, "total_gpm", "lbs_per_min"])
    if top_cols:
        st.subheader("Primary timeline")
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

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Detected events")
        st.dataframe(events_table, use_container_width=True, height=260)
    with c2:
        st.subheader("Event response summary")
        show_cols = [
            c for c in ["chemical", "event_type", "signal", "delta", "percent_change", "time_to_min", "persistence", "post_iqr", "n_events"]
            if c in event_metrics_df.columns
        ]
        st.dataframe(event_metrics_df[show_cols] if show_cols else event_metrics_df, use_container_width=True, height=260)


# --------------------------------------------------
# FULL TIMELINE
# --------------------------------------------------
elif page == "Full Timeline":
    st.title("Full Timeline")

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


# --------------------------------------------------
# EVENT WINDOWS
# --------------------------------------------------
elif page == "Event Windows":
    st.title("Event Windows")

    event_family = st.selectbox("Event family", list(EVENT_COLUMNS.keys()))
    event_direction = st.radio("Transition", ["ON", "OFF"], horizontal=True)
    signal_mode = st.radio("Window view", ["NH3 vs H2S", "NH3 vs Temperature", "H2S vs Temperature", "NH3 vs Load", "H2S vs Load"], horizontal=True)

    on_events, off_events = detect_transitions(master_df, EVENT_COLUMNS[event_family])
    event_times = on_events if event_direction == "ON" else off_events

    if len(event_times) == 0:
        st.warning("No events found for this selection.")
    else:
        event_time = st.selectbox("Event timestamp", list(event_times), format_func=lambda x: x.strftime("%Y-%m-%d %H:%M"))
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
    st.title("Event Study")

    c1, c2, c3 = st.columns(3)
    chem = c1.selectbox("Chemical", list(EVENT_COLUMNS.keys()))
    event_type = c2.selectbox("Event type", ["ON", "OFF"])
    signal_label = c3.selectbox("Signal", ["NH3", "H2S"])
    signal_col = NH3 if signal_label == "NH3" else H2S

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


# --------------------------------------------------
# TRANSITION COMPARISON
# --------------------------------------------------
elif page == "Transition Comparison":
    st.title("Transition Comparison")

    compare_options = {
        "NH3 vs H2S": (NH3, H2S, "NH₃ (ppm)", "H₂S (ppm)"),
        "NH3 vs Temperature": (NH3, TEMP_NH3, "NH₃ (ppm)", "Temperature (°F)"),
        "H2S vs Temperature": (H2S, TEMP_H2S, "H₂S (ppm)", "Temperature (°F)"),
        "NH3 vs Sludge Flow": (NH3, FLOW, "NH₃ (ppm)", "Sludge Flow (GPM)"),
        "H2S vs Sludge Flow": (H2S, FLOW, "H₂S (ppm)", "Sludge Flow (GPM)"),
    }
    choice = st.selectbox("Multi-panel view", list(compare_options.keys()))
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
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=list(ordered.keys()),
            specs=[[{"secondary_y": True}, {"secondary_y": True}], [{"secondary_y": True}, {"secondary_y": True}]],
        )
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        for (event_name, center), (r, c) in zip(ordered.items(), positions):
            w = master_df.loc[center - WINDOW_48H : center + WINDOW_48H].copy()
            if w.empty or y1 not in w.columns or y2 not in w.columns:
                continue
            w["minutes"] = (w.index - center).total_seconds() / 60
            fig.add_trace(go.Scatter(x=w["minutes"], y=w[y1], mode="lines", name=y1_label, showlegend=(r == 1 and c == 1)), row=r, col=c, secondary_y=False)
            fig.add_trace(go.Scatter(x=w["minutes"], y=w[y2], mode="lines", line=dict(dash="dot"), name=y2_label, showlegend=(r == 1 and c == 1)), row=r, col=c, secondary_y=True)
            fig.add_vline(x=0, line_dash="dash", line_color="black", row=r, col=c)
            fig.update_yaxes(title_text=y1_label, row=r, col=c, secondary_y=False)
            fig.update_yaxes(title_text=y2_label, row=r, col=c, secondary_y=True)
        fig.update_layout(title=f"{choice} Across Operational Transitions", template="plotly_white", hovermode="x unified", height=800, legend=dict(orientation="h"))
        fig.update_xaxes(title_text="Minutes from Event")
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
elif page == "Aggregates & Coverage":
    st.title("Aggregates & Coverage")

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


# --------------------------------------------------
# CORRELATION & LOAD ANALYSIS
# --------------------------------------------------
elif page == "Correlation & Load Analysis":
    st.title("Correlation & Load Analysis")

    analysis_df = daily_df.copy() if daily_df is not None else master_df.resample("1D").mean(numeric_only=True)
    analysis_df = analysis_df.copy()

    numeric_cols = [c for c in analysis_df.columns if pd.api.types.is_numeric_dtype(analysis_df[c])]
    default_corr = [c for c in [NH3, H2S, "total_gpm", "transferred_lbs_vol_daily", "transferred_lbs_vol", "nh3_std", "h2s_std", "ferric_active_lbs_per_day"] if c in numeric_cols]
    selected = st.multiselect("Columns for heatmap", numeric_cols, default=default_corr[: min(len(default_corr), 8)])
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
    st.plotly_chart(scatter_with_trend(scatter_source, x_col, y_col, color_col=color_col, title=f"{y_col} vs {x_col}"), use_container_width=True)

    norm_cols = available_columns(master_df, ["nh3_per_lb", "h2s_per_lb", "lbs_per_min", "total_gpm", NH3, H2S])
    if norm_cols:
        st.subheader("Normalized / load-aware variables")
        st.dataframe(master_df[norm_cols].dropna(how="all").tail(500), use_container_width=True, height=260)


# --------------------------------------------------
# ANOMALIES
# --------------------------------------------------
elif page == "Anomalies":
    st.title("Anomalies")

    candidates = available_columns(master_df, [NH3, H2S, RAW_NH3, RAW_H2S, TEMP_NH3, TEMP_H2S, "total_gpm", "lbs_per_min", "nh3_per_lb", "h2s_per_lb"])
    target_col = st.selectbox("Signal", candidates)
    window = st.slider("Rolling window (minutes)", min_value=60, max_value=4320, value=1440, step=60)
    threshold = st.slider("Absolute z-score threshold", min_value=2.0, max_value=6.0, value=3.0, step=0.25)

    z = add_zscore(master_df, target_col, window=window)
    anomalies = detect_anomalies(master_df, target_col, threshold=threshold, window=window)

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


# --------------------------------------------------
# DATA EXPLORER
# --------------------------------------------------
elif page == "Data Explorer":
    st.title("Data Explorer")

    dataset_name = st.selectbox("Dataset", ["master_1min", "master_1h", "master_daily", "monthly_summary", "weekday_summary", "event_metrics"])
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
            st.dataframe(display_df[show_cols].head(max_rows), use_container_width=True, height=500)

            if len(numeric_candidates) >= 1:
                st.subheader("Quick distribution")
                dist_col = st.selectbox("Numeric column", numeric_candidates)
                hist = go.Figure(data=[go.Histogram(x=view_df[dist_col].dropna(), nbinsx=50)])
                hist.update_layout(title=f"Distribution of {dist_col}", template="plotly_white")
                st.plotly_chart(hist, use_container_width=True)
