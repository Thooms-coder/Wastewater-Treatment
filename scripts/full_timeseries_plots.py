"""
Full-timeline odor and operations time-series plots (interactive Plotly version).

Generates dual-axis plots for:
1) NH3 + H2S
2) NH3 + Temperature
3) H2S + Temperature
4) H2S + Hourly Transferred Lbs Vol
5) NH3 + Hourly Transferred Lbs Vol

Transferred Lbs Vol:
TOTAL_GPM × 8.34 × 1.38 × 0.66
Aggregated hourly (hour ending timestamps)

Author: Mutsa Mungoshi
"""

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

from scripts.paths import PROCESSED_DATA_DIR


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures" / "full_timeseries"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = PROCESSED_DATA_DIR / "master_1min.parquet"


# ---------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------
NH3 = "nh3_roll_mean_15min"
H2S = "h2s_roll_max_15min"
TEMP_NH3 = "nh3_temperature_°f"
TEMP_H2S = "h2s_temperature_°f"

WEST_GPM = "west_sludge_out_gpm"
EAST_GPM = "east_sludge_out_gpm"
DIGESTER_GPM = "digesters_sludge_out_flow"

EVENT_COLUMNS = {
    "Ferric": "ferric_available",
    "HCl": "hcl_available",
}


# ---------------------------------------------------------------------
# Operational plant events (not ON/OFF transitions)
# ---------------------------------------------------------------------
PLANT_EVENTS = {
    "Ferric Reduced": pd.Timestamp("2026-01-07"),
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def safe_column(df, col):
    return col in df.columns and df[col].notna().any()


def detect_all_transitions(df):
    events = {}

    for chem_name, col in EVENT_COLUMNS.items():
        if col not in df.columns:
            continue

        diff = df[col].diff()

        on_times = df.index[diff == 1]
        off_times = df.index[diff == -1]

        events[f"{chem_name}_ON"] = list(on_times)
        events[f"{chem_name}_OFF"] = list(off_times)

    return events


def add_event_lines_plotly(fig, events, yref="paper"):
    # Chemical ON/OFF transitions
    for name, times in events.items():
        for t in times:
            fig.add_vline(
                x=t,
                line_dash="dash",
                line_width=1,
                opacity=0.35,
            )
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

    # Operational plant events
    for name, t in PLANT_EVENTS.items():
        fig.add_vline(
            x=t,
            line_dash="dot",
            line_color="purple",
            line_width=1.5,
        )
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


def simple_dual_axis_plot(
    df,
    y1_col,
    y2_col,
    y1_label,
    y2_label,
    title,
    fname,
    events
):
    if not safe_column(df, y1_col) or not safe_column(df, y2_col):
        return

    plot_df = df[[y1_col, y2_col]].dropna(how="all").copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df[y1_col],
            mode="lines",
            name=y1_label,
            line=dict(width=2),
            yaxis="y",
            hovertemplate=(
                "Time: %{x}<br>"
                f"{y1_label}: " + "%{y:.2f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df[y2_col],
            mode="lines",
            name=y2_label,
            line=dict(width=1.6, dash="dot"),
            yaxis="y2",
            hovertemplate=(
                "Time: %{x}<br>"
                f"{y2_label}: " + "%{y:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(title="Date / Time"),
        yaxis=dict(title=y1_label),
        yaxis2=dict(
            title=y2_label,
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h"),
    )

    add_event_lines_plotly(fig, events)
    fig.update_xaxes(rangeslider_visible=True)
    fig.write_html(FIG_DIR / fname)


def dual_axis_hourly_bar_plot(
    df,
    signal_col,
    hourly_lbs,
    signal_label,
    lbs_label,
    title,
    fname,
    events
):
    if not safe_column(df, signal_col):
        return

    plot_df = df[[signal_col]].dropna().copy()
    hourly_plot = hourly_lbs.dropna().copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df[signal_col],
            mode="lines",
            name=signal_label,
            line=dict(width=2),
            yaxis="y",
            hovertemplate=(
                "Time: %{x}<br>"
                f"{signal_label}: " + "%{y:.2f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Bar(
            x=hourly_plot.index,
            y=hourly_plot.values,
            name=lbs_label,
            yaxis="y2",
            hovertemplate=(
                "Hour ending: %{x}<br>"
                f"{lbs_label}: " + "%{y:,.0f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        barmode="overlay",
        xaxis=dict(title="Date / Time"),
        yaxis=dict(title=signal_label),
        yaxis2=dict(
            title=lbs_label,
            overlaying="y",
            side="right",
            tickformat="~s",
        ),
        legend=dict(orientation="h"),
    )

    add_event_lines_plotly(fig, events)
    fig.update_xaxes(rangeslider_visible=True)
    fig.write_html(FIG_DIR / fname)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run_full_timeseries_plots():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master dataset not found: {MASTER_PATH}")

    df = pd.read_parquet(MASTER_PATH).sort_index()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Master dataset index must be a DatetimeIndex.")

    events = detect_all_transitions(df)

    # -------------------------------------------------
    # Unified Sludge Flow Logic
    # -------------------------------------------------
    for col in [WEST_GPM, EAST_GPM, DIGESTER_GPM]:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0)
        else:
            df[col] = 0

    df["total_gpm"] = (
        df[WEST_GPM]
        + df[EAST_GPM]
        + df[DIGESTER_GPM]
    )

    # Excel constant
    k = 8.34 * 1.38 * 0.66

    df["lbs_per_min"] = df["total_gpm"] * k

    # Hour ending timestamps
    hourly_lbs = (
        df["lbs_per_min"]
        .resample("1h", label="right", closed="right")
        .sum()
    )

    # -----------------------------
    # NEW: Hourly gas signals
    # -----------------------------
    hourly_signals = df.resample("1h").agg({
        NH3: "mean",
        H2S: "max",
        TEMP_NH3: "mean",
        TEMP_H2S: "mean"
    })

    hourly_signals = hourly_signals.dropna(how="all")

    hourly_lbs = hourly_lbs[hourly_lbs > 0]

    # -------------------------------------------------
    # Plots
    # -------------------------------------------------
    simple_dual_axis_plot(
        df,
        NH3,
        H2S,
        "NH₃ (ppm)",
        "H₂S (ppm)",
        "NH₃ and H₂S – Full Timeline",
        "nh3_h2s_full.html",
        events
    )

    # -----------------------------
    # NEW: Hourly NH3 vs H2S
    # -----------------------------
    simple_dual_axis_plot(
        hourly_signals,
        NH3,
        H2S,
        "NH₃ (ppm) — hourly avg",
        "H₂S (ppm) — hourly max",
        "NH₃ and H₂S – Hourly",
        "nh3_h2s_hourly.html",
        events
    )

    simple_dual_axis_plot(
        df,
        NH3,
        TEMP_NH3,
        "NH₃ (ppm)",
        "Temperature (°F)",
        "NH₃ and Temperature – Full Timeline",
        "nh3_temp_full.html",
        events
    )

    simple_dual_axis_plot(
        hourly_signals,
        NH3,
        TEMP_NH3,
        "NH₃ (ppm) — hourly avg",
        "Temperature (°F) — hourly avg",
        "NH₃ and Temperature – Hourly",
        "nh3_temp_hourly.html",
        events
    )

    simple_dual_axis_plot(
        df,
        H2S,
        TEMP_H2S,
        "H₂S (ppm)",
        "Temperature (°F)",
        "H₂S and Temperature – Full Timeline",
        "h2s_temp_full.html",
        events
    )

    simple_dual_axis_plot(
        hourly_signals,
        H2S,
        TEMP_H2S,
        "H₂S (ppm) — hourly max",
        "Temperature (°F) — hourly avg",
        "H₂S and Temperature – Hourly",
        "h2s_temp_hourly.html",
        events
    )

    dual_axis_hourly_bar_plot(
        df,
        H2S,
        hourly_lbs,
        "H₂S (ppm)",
        "Transferred Lbs Vol",
        "H₂S & Hourly Transferred Lbs Vol – Full Timeline",
        "h2s_transferred_lbs_vol_full.html",
        events
    )

    dual_axis_hourly_bar_plot(
        df,
        NH3,
        hourly_lbs,
        "NH₃ (ppm)",
        "Transferred Lbs Vol",
        "NH₃ & Hourly Transferred Lbs Vol – Full Timeline",
        "nh3_transferred_lbs_vol_full.html",
        events
    )

    print(f"✓ Full timeline interactive figures saved to:\n{FIG_DIR}")

    print("\n===== MASTER DATE RANGE =====")
    print("Min:", df.index.min())
    print("Max:", df.index.max())

    print("\nLast 10 timestamps:")
    print(df.tail(10))


if __name__ == "__main__":
    run_full_timeseries_plots()