"""
Multi-panel comparison figures for odor and operations data (Plotly).

Creates 4-panel plots (2x2) for each variable pair across:
- Ferric OFF
- Ferric ON
- HCl OFF
- HCl ON

Each panel shows ±48 hours around the first occurrence
of each transition type.

Author: Mutsa Mungoshi
"""

from pathlib import Path
import pandas as pd

from scripts.analytics import detect_transitions
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
from scripts.plotting import multi_panel_figure


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures" / "multi_panel"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = PROCESSED_DATA_DIR / "master_1min.parquet"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def safe_column(df, col):
    return col in df.columns and df[col].notna().any()


def detect_first_transitions(df):
    events = {}

    for chem_name, col in EVENT_COLUMNS.items():
        on_times, off_times = detect_transitions(df, col)

        if not off_times.empty:
            events[f"{chem_name} OFF"] = off_times[0]

        if not on_times.empty:
            events[f"{chem_name} ON"] = on_times[0]

    ordered = {}
    for key in ["Ferric OFF", "Ferric ON", "HCl OFF", "HCl ON"]:
        if key in events:
            ordered[key] = events[key]

    return ordered


def extract_window(df, center):
    w = df.loc[center - WINDOW : center + WINDOW].copy()
    if w.empty:
        return w
    w["minutes"] = (w.index - center).total_seconds() / 60
    return w


def multi_panel(df, events, y1, y2, y1_label, y2_label, title, fname):
    if not safe_column(df, y1) or not safe_column(df, y2):
        return

    event_windows = {event_name: extract_window(df, t) for event_name, t in events.items()}
    fig = multi_panel_figure(df, event_windows, y1, y2, y1_label, y2_label, title)
    fig.write_html(FIG_DIR / fname)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run_multi_panel_plots():

    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master dataset not found: {MASTER_PATH}")

    df = pd.read_parquet(MASTER_PATH).sort_index()

    events = detect_first_transitions(df)

    if len(events) < 4:
        print("[WARN] Fewer than four transition types detected.")

    multi_panel(
        df, events,
        NH3, H2S,
        "NH₃ (ppm)", "H₂S (ppm)",
        "NH₃ vs H₂S Across Operational Transitions",
        "nh3_h2s_multipanel.html"
    )

    multi_panel(
        df, events,
        NH3, TEMP_NH3,
        "NH₃ (ppm)", "Temperature (°F)",
        "NH₃ vs Temperature Across Operational Transitions",
        "nh3_temp_multipanel.html"
    )

    multi_panel(
        df, events,
        H2S, TEMP_H2S,
        "H₂S (ppm)", "Temperature (°F)",
        "H₂S vs Temperature Across Operational Transitions",
        "h2s_temp_multipanel.html"
    )

    multi_panel(
        df, events,
        NH3, FLOW,
        "NH₃ (ppm)", "Sludge Flow (GPM)",
        "NH₃ vs Sludge Flow Across Operational Transitions",
        "nh3_flow_multipanel.html"
    )

    multi_panel(
        df, events,
        H2S, FLOW,
        "H₂S (ppm)", "Sludge Flow (GPM)",
        "H₂S vs Sludge Flow Across Operational Transitions",
        "h2s_flow_multipanel.html"
    )

    print(f"✓ Multi-panel interactive figures saved to:\n{FIG_DIR}")


if __name__ == "__main__":
    run_multi_panel_plots()
