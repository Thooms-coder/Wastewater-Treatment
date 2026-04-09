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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.paths import PROCESSED_DATA_DIR


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures" / "multi_panel"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = PROCESSED_DATA_DIR / "master_1min.parquet"


# ---------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------
NH3 = "nh3_roll_mean_15min"
H2S = "h2s_roll_max_15min"
TEMP_NH3 = "nh3_temperature_°f"
TEMP_H2S = "h2s_temperature_°f"
FLOW = "east_sludge_out_gpm_combined"

EVENT_COLUMNS = {
    "Ferric": "ferric_available",
    "HCl": "hcl_available",
}

WINDOW = pd.Timedelta(hours=48)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def safe_column(df, col):
    return col in df.columns and df[col].notna().any()


def detect_first_transitions(df):
    events = {}

    for chem_name, col in EVENT_COLUMNS.items():

        if col not in df.columns:
            continue

        diff = df[col].diff()

        off_times = df.index[diff == -1]
        on_times = df.index[diff == 1]

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


# ---------------------------------------------------------------------
# 🔥 Plotly Multi-Panel
# ---------------------------------------------------------------------
def multi_panel(df, events, y1, y2,
                y1_label, y2_label,
                title, fname):

    if not safe_column(df, y1) or not safe_column(df, y2):
        return

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=list(events.keys()),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}]]
    )

    positions = [(1,1), (1,2), (2,1), (2,2)]

    for (event_name, t), (r, c) in zip(events.items(), positions):

        w = extract_window(df, t)

        if w.empty:
            continue

        # Primary signal
        fig.add_trace(
            go.Scatter(
                x=w["minutes"],
                y=w[y1],
                mode="lines",
                name=y1_label,
                line=dict(width=2),
                showlegend=(r == 1 and c == 1),
                hovertemplate=f"{y1_label}: " + "%{y:.2f}<br>Δmin: %{x}<extra></extra>",
            ),
            row=r, col=c,
            secondary_y=False
        )

        # Secondary signal
        fig.add_trace(
            go.Scatter(
                x=w["minutes"],
                y=w[y2],
                mode="lines",
                name=y2_label,
                line=dict(dash="dot"),
                showlegend=(r == 1 and c == 1),
                hovertemplate=f"{y2_label}: " + "%{y:.2f}<br>Δmin: %{x}<extra></extra>",
            ),
            row=r, col=c,
            secondary_y=True
        )

        # Event marker
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="black",
            row=r,
            col=c
        )

        # Axis labels
        fig.update_yaxes(title_text=y1_label, row=r, col=c, secondary_y=False)
        fig.update_yaxes(title_text=y2_label, row=r, col=c, secondary_y=True)

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        height=800,
        legend=dict(orientation="h"),
    )

    fig.update_xaxes(title_text="Minutes from Event")
    fig.update_xaxes(rangeslider_visible=True)
    
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