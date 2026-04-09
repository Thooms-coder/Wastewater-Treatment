"""
Event-window time-series visualizations for odor analysis (INTERACTIVE).

Generates Plotly-based multi-axis interactive plots for NH3 and H2S
around detected operational transitions (Ferric / HCl ON/OFF).

Each window: ±48 hours.

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
FIG_DIR = PROJECT_ROOT / "figures" / "event_windows"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = PROCESSED_DATA_DIR / "master_1min.parquet"

WINDOW = pd.Timedelta(hours=48)

NH3 = "nh3_roll_mean_15min"
H2S = "h2s_roll_max_15min"
NH3_TEMP = "nh3_temperature_°f"
H2S_TEMP = "h2s_temperature_°f"
FLOW = "east_sludge_out_gpm_combined"

EVENT_COLUMNS = {
    "Ferric": "ferric_available",
    "HCl": "hcl_available",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def detect_transitions(df):
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


def extract_window(df, center_time):
    w = df.loc[center_time - WINDOW : center_time + WINDOW].copy()
    if w.empty:
        return w

    w["minutes_from_event"] = (
        (w.index - center_time).total_seconds() / 60
    )
    return w


def safe_column(df, col):
    return col in df.columns and df[col].notna().any()


# ---------------------------------------------------------------------
# 🔥 NEW: Plotly Dual Axis Plot
# ---------------------------------------------------------------------
def dual_axis_plot_plotly(df, y1, y2,
                         y1_label, y2_label,
                         title, fname,
                         bar=False):

    fig = go.Figure()

    # ---------------- Primary axis ----------------
    fig.add_trace(
        go.Scatter(
            x=df["minutes_from_event"],
            y=df[y1],
            mode="lines",
            name=y1_label,
            line=dict(width=2),
            customdata=df.index,
            hovertemplate="Time: %{customdata}<br>"
                          + f"{y1_label}: " + "%{y:.2f}<br>"
                          + "Δmin: %{x}<extra></extra>",
        )
    )

    # ---------------- Secondary axis ----------------
    if bar:
        fig.add_trace(
            go.Bar(
                x=df["minutes_from_event"],
                y=df[y2],
                name=y2_label,
                opacity=0.6,
                yaxis="y2",
                hovertemplate=f"{y2_label}: " + "%{y:.2f}<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["minutes_from_event"],
                y=df[y2],
                mode="lines",
                name=y2_label,
                yaxis="y2",
                line=dict(dash="dot"),
                hovertemplate=f"{y2_label}: " + "%{y:.2f}<extra></extra>",
            )
        )

    # ---------------- Layout ----------------
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(title="Minutes from Event"),
        yaxis=dict(title=y1_label),
        yaxis2=dict(
            title=y2_label,
            overlaying="y",
            side="right"
        ),
        legend=dict(orientation="h"),
    )

    # Event marker
    fig.add_vline(x=0, line_dash="dash", line_color="black")

    # Save interactive HTML
    fig.write_html(FIG_DIR / fname.replace(".png", ".html"))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run_event_window_plots():

    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master dataset not found: {MASTER_PATH}")

    df = pd.read_parquet(MASTER_PATH).sort_index()

    # -------------------------------------------------
    # Volatile mass transfer
    # -------------------------------------------------
    if FLOW in df.columns:
        df[FLOW] = df[FLOW].ffill().fillna(0)
        total_gph = df[FLOW] * 60

        DENSITY = 8.34
        SOLIDS_FACTOR = 1.38
        VOLATILE_FRAC = 0.66

        df["transferred_lbs_vol"] = (
            total_gph * DENSITY * SOLIDS_FACTOR * VOLATILE_FRAC
        )
    else:
        df["transferred_lbs_vol"] = 0

    events = detect_transitions(df)

    for event_name, times in events.items():

        if not times:
            continue

        for event_time in times:

            window_df = extract_window(df, event_time)

            if window_df.empty:
                continue

            base = f"{event_name}_{event_time.strftime('%Y%m%d_%H%M')}"

            # NH3 + H2S
            if safe_column(window_df, NH3) and safe_column(window_df, H2S):
                dual_axis_plot_plotly(
                    window_df,
                    NH3,
                    H2S,
                    "NH₃ (ppm)",
                    "H₂S (ppm)",
                    f"NH₃ and H₂S Around {event_name}",
                    f"{base}_nh3_h2s.html"
                )

            # H2S + Temperature
            if safe_column(window_df, H2S) and safe_column(window_df, H2S_TEMP):
                dual_axis_plot_plotly(
                    window_df,
                    H2S,
                    H2S_TEMP,
                    "H₂S (ppm)",
                    "Temperature (°F)",
                    f"H₂S and Temperature Around {event_name}",
                    f"{base}_h2s_temp.html"
                )

            # NH3 + Temperature
            if safe_column(window_df, NH3) and safe_column(window_df, NH3_TEMP):
                dual_axis_plot_plotly(
                    window_df,
                    NH3,
                    NH3_TEMP,
                    "NH₃ (ppm)",
                    "Temperature (°F)",
                    f"NH₃ and Temperature Around {event_name}",
                    f"{base}_nh3_temp.html"
                )

            # H2S + Volatile Load
            if safe_column(window_df, H2S):
                dual_axis_plot_plotly(
                    window_df,
                    H2S,
                    "transferred_lbs_vol",
                    "H₂S (ppm)",
                    "Transferred Vol (lbs/min equiv)",
                    f"H₂S & Volatile Load Around {event_name}",
                    f"{base}_h2s_lbs.html",
                    bar=True
                )

            # NH3 + Volatile Load
            if safe_column(window_df, NH3):
                dual_axis_plot_plotly(
                    window_df,
                    NH3,
                    "transferred_lbs_vol",
                    "NH₃ (ppm)",
                    "Transferred Vol (lbs/min equiv)",
                    f"NH₃ & Volatile Load Around {event_name}",
                    f"{base}_nh3_lbs.html",
                    bar=True
                )

            print(f"✓ Finished plots for {event_name} at {event_time}")

    print(f"\nAll interactive figures written to: {FIG_DIR}")


if __name__ == "__main__":
    run_event_window_plots()
