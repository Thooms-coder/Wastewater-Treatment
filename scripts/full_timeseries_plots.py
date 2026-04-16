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

from scripts.analytics import add_operational_features, detect_all_transitions
from scripts.constants import (
    DIGESTER_GPM,
    EAST_GPM,
    EVENT_COLUMNS,
    H2S,
    NH3,
    PLANT_EVENTS,
    TEMP_H2S,
    TEMP_NH3,
    WEST_GPM,
)
from scripts.paths import PROCESSED_DATA_DIR
from scripts.plotting import dual_axis_figure


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures" / "full_timeseries"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = PROCESSED_DATA_DIR / "master_1min.parquet"


def safe_column(df, col):
    return col in df.columns and df[col].notna().any()


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

    fig = dual_axis_figure(
        plot_df,
        y1_col,
        y2_col,
        y1_label,
        y2_label,
        title,
        add_events=events,
        plant_events=PLANT_EVENTS,
        margin=dict(l=40, r=40, t=60, b=40),
    )
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

    merged = plot_df.join(hourly_plot.rename(lbs_label), how="outer")
    fig = dual_axis_figure(
        merged,
        signal_col,
        lbs_label,
        signal_label,
        lbs_label,
        title,
        add_events=events,
        plant_events=PLANT_EVENTS,
        bar_second=True,
        margin=dict(l=40, r=40, t=60, b=40),
        secondary_tickformat="~s",
    )
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

    events = detect_all_transitions(df, EVENT_COLUMNS)
    df = add_operational_features(df)

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
