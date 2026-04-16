"""
Multi-event quantitative metrics for odor response.

Purpose:
--------
Numerically summarize NH3 and H2S responses to operational transitions
(Ferric ON/OFF, HCl ON/OFF).

Enhancements:
-------------
- Robust NaN handling
- Additional variability metrics (IQR)
- Safer time-to-min computation
- Improved diagnostics

Author: Mutsa Mungoshi
"""

import pandas as pd

from scripts.analytics import (
    BASELINE_WINDOW,
    POST_WINDOW,
    aggregate_event_metrics,
    compute_single_event_metrics,
    detect_transitions,
    extract_relative_series,
    window_slice,
)
from scripts.constants import EVENT_COLUMNS, H2S, NH3
from scripts.paths import PROCESSED_DATA_DIR


# --------------------------------------------------
# Configuration
# --------------------------------------------------
MASTER_FILE = PROCESSED_DATA_DIR / "master_1min.parquet"

TARGETS = {
    "NH3": NH3,
    "H2S": H2S,
}

# --------------------------------------------------
# Main workflow
# --------------------------------------------------
def run_event_metrics():

    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"Master dataset not found: {MASTER_FILE}")

    df = pd.read_parquet(MASTER_FILE).sort_index()

    results = []

    for chem_name, event_col in EVENT_COLUMNS.items():

        if event_col not in df.columns:
            print(f"[WARN] Missing column: {event_col}")
            continue

        on_events, off_events = detect_transitions(df, event_col)

        for event_type, event_times in {
            "ON": on_events,
            "OFF": off_events,
        }.items():

            for signal_name, column in TARGETS.items():

                if column not in df.columns:
                    print(f"[WARN] Missing signal column: {column}")
                    continue

                event_metrics = []

                for event_time in event_times:

                    series = extract_relative_series(df, event_time, column)

                    if series is None:
                        continue

                    baseline = window_slice(series, BASELINE_WINDOW)
                    post     = window_slice(series, POST_WINDOW)

                    if baseline.empty or post.empty:
                        continue

                    metrics = compute_single_event_metrics(
                        baseline,
                        post,
                        post_label="post_median",
                        time_to_min_label="time_to_min_min",
                        persistence_label="persistence_min",
                    )
                    event_metrics.append(metrics)

                if not event_metrics:
                    print(f"[WARN] No valid events for {chem_name} {event_type} ({signal_name})")
                    continue

                agg_metrics = aggregate_event_metrics(
                    event_metrics,
                    post_label="post_median",
                    time_to_min_label="time_to_min_min",
                    persistence_label="persistence_min",
                )

                # --------------------------------------------------
                # Domain sanity checks (important for your research)
                # --------------------------------------------------
                if chem_name == "Ferric" and event_type == "OFF":
                    if agg_metrics["delta"] < 0:
                        print(f"[WARN] Unexpected decrease after Ferric OFF ({signal_name})")

                if chem_name == "Ferric" and event_type == "ON":
                    if agg_metrics["delta"] > 0:
                        print(f"[WARN] Unexpected increase after Ferric ON ({signal_name})")

                if chem_name == "HCl" and event_type == "ON" and signal_name == "NH3":
                    if agg_metrics["delta"] > 0:
                        print(f"[WARN] NH3 increased after HCl ON")

                results.append({
                    "chemical": chem_name,
                    "event_type": event_type,
                    "signal": signal_name,
                    **agg_metrics
                })

                print(
                    f"✓ {chem_name} {event_type} | {signal_name} "
                    f"(n={agg_metrics['n_events']})"
                )

    return pd.DataFrame(results)


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":

    metrics_df = run_event_metrics()

    print("\nMulti-event response metrics:")
    print(metrics_df.round(2))

    out_path = PROCESSED_DATA_DIR / "event_metrics.csv"
    metrics_df.to_csv(out_path, index=False)

    print("\nMetrics saved to:", out_path)
