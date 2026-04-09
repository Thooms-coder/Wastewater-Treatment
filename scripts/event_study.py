"""
Multi-event study analysis for odor response around operational changes.

Purpose:
--------
Quantify and visualize NH3 and H2S behavior before and after
operational transitions (Ferric ON/OFF, HCl ON/OFF).

Method:
-------
- Automatically detects state transitions
- Aligns time series around each event
- Aggregates across events using median + IQR
- Performs pre-trend diagnostic checks
- Outputs interactive Plotly visualizations

Author: Mutsa Mungoshi
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scripts.paths import PROCESSED_DATA_DIR, EVENT_STUDY_PLOTS_DIR


# --------------------------------------------------
# Configuration
# --------------------------------------------------
MASTER_FILE = PROCESSED_DATA_DIR / "master_1min.parquet"

TARGETS = {
    "h2s": "h2s_roll_max_15min",
    "nh3": "nh3_roll_mean_15min",
}

EVENT_COLUMNS = {
    "ferric": "ferric_available",
    "hcl": "hcl_available",
}

WINDOW_MINUTES = 72 * 60  # ±72 hours
PRETREND_WINDOW = (-1440, -60)  # -24h to -1h
PRETREND_TOL = 0.1


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def detect_transitions(df, column):
    diff = df[column].diff()
    on_events = df.index[diff == 1]
    off_events = df.index[diff == -1]
    return on_events, off_events


def extract_event_window(df, event_time, column, window):
    start = event_time - pd.Timedelta(minutes=window)
    end = event_time + pd.Timedelta(minutes=window)

    subset = df.loc[start:end, column].copy()

    if subset.empty:
        return None

    aligned_index = (
        (subset.index - event_time)
        .total_seconds() / 60
    ).astype(int)

    subset.index = aligned_index

    return subset


def summarize_event(aligned_df):
    return pd.DataFrame({
        "median": aligned_df.median(axis=1),
        "q25": aligned_df.quantile(0.25, axis=1),
        "q75": aligned_df.quantile(0.75, axis=1),
    })


def check_pretrend(summary, window, tolerance):
    pre = summary.loc[window[0]:window[1], "median"]

    if pre.empty:
        return True

    variation = pre.max() - pre.min()
    return variation <= tolerance


# --------------------------------------------------
# 🔥 Plotly Visualization
# --------------------------------------------------
def plot_event(summary, title, ylabel, output_path):

    fig = go.Figure()

    # ---------------- IQR band ----------------
    fig.add_trace(
        go.Scatter(
            x=summary.index,
            y=summary["q75"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

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

    # ---------------- Median ----------------
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

    # ---------------- Event marker ----------------
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Event",
        annotation_position="top"
    )

    # ---------------- Layout ----------------
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Minutes from event",
        yaxis_title=ylabel,
        legend=dict(orientation="h"),
    )

    fig.update_xaxes(rangeslider_visible=True)
    # ---------------- Save ----------------
    fig.write_html(output_path.with_suffix(".html"))


# --------------------------------------------------
# Main workflow
# --------------------------------------------------
def run_event_study():

    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"Master dataset not found: {MASTER_FILE}")

    df = pd.read_parquet(MASTER_FILE).sort_index()

    for target_name, target_col in TARGETS.items():

        if target_col not in df.columns:
            print(f"[WARN] Missing column: {target_col}")
            continue

        for chem_name, event_col in EVENT_COLUMNS.items():

            if event_col not in df.columns:
                print(f"[WARN] Missing event column: {event_col}")
                continue

            on_events, off_events = detect_transitions(df, event_col)

            for event_type, event_times in {
                "ON": on_events,
                "OFF": off_events,
            }.items():

                aligned_windows = []

                for t in event_times:
                    window = extract_event_window(
                        df,
                        t,
                        target_col,
                        WINDOW_MINUTES
                    )
                    if window is not None:
                        aligned_windows.append(window)

                if not aligned_windows:
                    print(f"[WARN] No valid windows for {chem_name} {event_type}")
                    continue

                aligned_df = pd.concat(aligned_windows, axis=1)

                summary = summarize_event(aligned_df)

                if not check_pretrend(summary, PRETREND_WINDOW, PRETREND_TOL):
                    print(
                        f"[WARN] Pre-trend instability detected for "
                        f"{target_name.upper()} around "
                        f"{chem_name.upper()} {event_type}"
                    )

                output_path = (
                    EVENT_STUDY_PLOTS_DIR /
                    f"{target_name}_{chem_name}_{event_type}.html"
                )

                plot_event(
                    summary,
                    title=(
                        f"{target_name.upper()} Response Around "
                        f"{chem_name.upper()} {event_type}"
                    ),
                    ylabel=f"{target_name.upper()} (ppm)",
                    output_path=output_path
                )

                print(
                    f"✓ {target_name.upper()} — "
                    f"{chem_name.upper()} {event_type} "
                    f"(n={aligned_df.shape[1]} events)"
                )


if __name__ == "__main__":
    run_event_study()
    print("Interactive event study plots written to:", EVENT_STUDY_PLOTS_DIR)
