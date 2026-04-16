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

from scripts.analytics import (
    check_pretrend,
    detect_transitions,
    extract_event_window,
    summarize_event,
)
from scripts.constants import (
    EVENT_COLUMNS_LOWER,
    EVENT_STUDY_WINDOW,
    H2S,
    NH3,
    PRETREND_TOL,
    PRETREND_WINDOW,
)
from scripts.paths import PROCESSED_DATA_DIR, EVENT_STUDY_PLOTS_DIR
from scripts.plotting import event_study_figure


# --------------------------------------------------
# Configuration
# --------------------------------------------------
MASTER_FILE = PROCESSED_DATA_DIR / "master_1min.parquet"

TARGETS = {
    "h2s": H2S,
    "nh3": NH3,
}
EVENT_COLUMNS = EVENT_COLUMNS_LOWER
WINDOW_MINUTES = EVENT_STUDY_WINDOW


def plot_event(summary, title, ylabel, output_path):
    fig = event_study_figure(summary, title, ylabel)
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
                    window = extract_event_window(df, t, target_col, WINDOW_MINUTES)
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
