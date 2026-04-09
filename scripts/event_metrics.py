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
import numpy as np

from scripts.paths import PROCESSED_DATA_DIR


# --------------------------------------------------
# Configuration
# --------------------------------------------------
MASTER_FILE = PROCESSED_DATA_DIR / "master_1min.parquet"

TARGETS = {
    "NH3": "nh3_roll_mean_15min",
    "H2S": "h2s_roll_max_15min",
}

EVENT_COLUMNS = {
    "Ferric": "ferric_available",
    "HCl": "hcl_available",
}

BASELINE_WINDOW = (-48 * 60, -12 * 60)
POST_WINDOW     = ( 12 * 60,  96 * 60)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def detect_transitions(df, column):
    diff = df[column].diff()
    on_events  = df.index[diff == 1]
    off_events = df.index[diff == -1]
    return on_events, off_events


def extract_relative_series(df, event_time, column):
    if column not in df.columns:
        return None

    series = df[column].dropna()

    if series.empty:
        return None

    rel_time = (series.index - event_time).total_seconds() / 60
    series.index = rel_time.astype(int)

    return series


def window_slice(series, window):
    if series is None:
        return pd.Series(dtype=float)

    return series.loc[
        (series.index >= window[0]) &
        (series.index <= window[1])
    ]


def compute_single_event_metrics(baseline, post):

    baseline_val = baseline.median()
    post_val     = post.median()

    # --- Core effect ---
    delta = post_val - baseline_val

    pct_change = (
        (delta / baseline_val * 100)
        if baseline_val not in [0, np.nan] else np.nan
    )

    # --- Time to minimum (safe) ---
    if post.empty:
        time_to_min = np.nan
    else:
        time_to_min = post.idxmin()

    # --- Persistence below baseline ---
    below = post[post < baseline_val]

    if below.empty:
        persistence = 0
    else:
        persistence = below.index.max() - below.index.min()

    # --- Variability (new, important) ---
    iqr_post = post.quantile(0.75) - post.quantile(0.25)

    return {
        "baseline": baseline_val,
        "post_median": post_val,
        "delta": delta,
        "percent_change": pct_change,
        "time_to_min_min": time_to_min,
        "persistence_min": persistence,
        "post_iqr": iqr_post,
    }


def aggregate_event_metrics(event_metrics_list):

    df = pd.DataFrame(event_metrics_list)

    return {
        "baseline": df["baseline"].median(),
        "post_median": df["post_median"].median(),
        "delta": df["delta"].median(),
        "percent_change": df["percent_change"].median(),
        "time_to_min_min": df["time_to_min_min"].median(),
        "persistence_min": df["persistence_min"].median(),
        "post_iqr": df["post_iqr"].median(),
        "n_events": len(df),
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

                    metrics = compute_single_event_metrics(baseline, post)
                    event_metrics.append(metrics)

                if not event_metrics:
                    print(f"[WARN] No valid events for {chem_name} {event_type} ({signal_name})")
                    continue

                agg_metrics = aggregate_event_metrics(event_metrics)

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
