"""
Event-window time-series visualizations for odor analysis (INTERACTIVE).

Generates Plotly-based multi-axis interactive plots for NH3 and H2S
around detected operational transitions (Ferric / HCl ON/OFF).

Each window: ±48 hours.

Author: Mutsa Mungoshi
"""

from pathlib import Path
import pandas as pd

from scripts.analytics import detect_all_transitions
from scripts.constants import (
    EVENT_COLUMNS,
    FLOW,
    H2S,
    NH3,
    TEMP_H2S,
    TEMP_NH3,
    WINDOW_48H as WINDOW,
)
from scripts.paths import PROCESSED_DATA_DIR
from scripts.plotting import event_window_figure


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures" / "event_windows"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = PROCESSED_DATA_DIR / "master_1min.parquet"
NH3_TEMP = TEMP_NH3
H2S_TEMP = TEMP_H2S


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
def dual_axis_plot_plotly(df, y1, y2, y1_label, y2_label, title, fname, bar=False):
    fig = event_window_figure(df, y1, y2, y1_label, y2_label, title, bar=bar)
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

    events = detect_all_transitions(df, EVENT_COLUMNS)

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
