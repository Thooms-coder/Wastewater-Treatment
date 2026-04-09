# events.py

"""
Detect chemical ON/OFF transitions and save structured events table.

Operational notes encoded:
- HCl stopped Aug 27 2025 (empty tank)
- HCl resumed Oct 2 2025 @ 14:30 after delivery
- HCl spill Nov 6 2025 → no HCl feeding afterward
- Ferric stopped Sept 9 2025 @ 08:00 (tote empty)
- Ferric restarted Sept 17 2025 @ 12:00
- Ferric feed reduced Jan 7 2026
- NH3/H2S sensors removed Oct 8 2025 @ 08:00
"""

import sys
from pathlib import Path

from config import PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT))

import pandas as pd


# ---------------------------------------------------------------------
# Flag Construction
# ---------------------------------------------------------------------
def add_event_flags(df):
    df = df.copy()

    df["hcl_available"] = 1
    df["ferric_available"] = 1
    df["sensor_valid"] = 1
    df["truck_activity_proxy"] = 0
    df["ferric_reduced_rate"] = 0   # new operational flag

    # ---------------- HCl downtime ----------------
    hcl_off_start = pd.Timestamp("2025-08-27")
    hcl_on_resume = pd.Timestamp("2025-10-02 14:30")

    df.loc[
        (df.index >= hcl_off_start) &
        (df.index < hcl_on_resume),
        "hcl_available"
    ] = 0

    # ---------------- HCl spill shutdown ----------------
    # Nov 6 spill → HCl no longer pumped
    hcl_spill = pd.Timestamp("2025-11-06")

    df.loc[df.index >= hcl_spill, "hcl_available"] = 0

    # ---------------- Ferric downtime ----------------
    ferric_off_start = pd.Timestamp("2025-09-09 08:00")
    ferric_on_resume = pd.Timestamp("2025-09-17 12:00")

    df.loc[
        (df.index >= ferric_off_start) &
        (df.index < ferric_on_resume),
        "ferric_available"
    ] = 0

    # ---------------- Ferric reduced feed ----------------
    ferric_reduction = pd.Timestamp("2026-01-07")

    df.loc[df.index >= ferric_reduction, "ferric_reduced_rate"] = 1

    # ---------------- Sensor validity ----------------
    sensor_off_time = pd.Timestamp("2025-10-08 08:00")

    df.loc[df.index >= sensor_off_time, "sensor_valid"] = 0

    # ---------------- Truck proxy ----------------
    weekday_mask = df.index.weekday < 5
    daytime_mask = (df.index.hour >= 7) & (df.index.hour <= 18)

    df.loc[weekday_mask & daytime_mask, "truck_activity_proxy"] = 1

    return df


# ---------------------------------------------------------------------
# Transition Detection
# ---------------------------------------------------------------------
def detect_transitions(df, column, chemical_name):
    """
    Detect ON and OFF transitions for a binary availability column.
    Uses robust state-change detection.
    """

    state_change = df[column].ne(df[column].shift())

    on_events = df.index[state_change & (df[column] == 1)]
    off_events = df.index[state_change & (df[column] == 0)]

    records = []

    for ts in on_events:
        records.append({
            "timestamp": ts,
            "chemical": chemical_name,
            "event_type": "ON"
        })

    for ts in off_events:
        records.append({
            "timestamp": ts,
            "chemical": chemical_name,
            "event_type": "OFF"
        })

    return records


def build_events_table(df):
    records = []

    records += detect_transitions(df, "hcl_available", "HCl")
    records += detect_transitions(df, "ferric_available", "Ferric")

    events = pd.DataFrame(records).sort_values("timestamp")

    return events


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from scripts.load_data import load_all_data
    from scripts.preprocess import preprocess_data
    from scripts.features import build_features

    raw = load_all_data()
    clean = preprocess_data(raw)
    features, targets, derived = build_features(clean)

    df = add_event_flags(features)

    events = build_events_table(df)

    # Save to disk
    output_path = Path("data/processed/events.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    events.to_csv(output_path, index=False)

    print("\nDetected Events:\n")
    print(events)

    print("\nEvent counts:\n")
    print(events.groupby(["chemical", "event_type"]).size())

    print(f"\n✓ Events saved to: {output_path.resolve()}")