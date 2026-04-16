import pandas as pd


# --------------------------------------------------
# Core signal columns
# --------------------------------------------------
NH3 = "nh3_roll_mean_15min"
H2S = "h2s_roll_max_15min"
RAW_NH3 = "nh3_nh3_ppm"
RAW_H2S = "h2s_h2s_ppm"
TEMP_NH3 = "nh3_temperature_°f"
TEMP_H2S = "h2s_temperature_°f"


# --------------------------------------------------
# Operational flow columns
# --------------------------------------------------
WEST_GPM = "west_sludge_out_gpm"
EAST_GPM = "east_sludge_out_gpm"
DIGESTER_GPM = "digesters_sludge_out_flow"
DIG_GPM = DIGESTER_GPM
FLOW = "east_sludge_out_gpm_combined"

FLOW_COLS = [
    WEST_GPM,
    EAST_GPM,
    DIGESTER_GPM,
]

WATER_COLS = [
    EAST_GPM,
    WEST_GPM,
    DIGESTER_GPM,
    FLOW,
]


# --------------------------------------------------
# Event configuration
# --------------------------------------------------
EVENT_COLUMNS = {
    "Ferric": "ferric_available",
    "HCl": "hcl_available",
}

EVENT_COLUMNS_LOWER = {
    "ferric": "ferric_available",
    "hcl": "hcl_available",
}

PLANT_EVENTS = {
    "Ferric Reduced": pd.Timestamp("2026-01-07"),
}


# --------------------------------------------------
# Analysis windows
# --------------------------------------------------
BASELINE_WINDOW = (-48 * 60, -12 * 60)
POST_WINDOW = (12 * 60, 96 * 60)
EVENT_STUDY_WINDOW = 72 * 60
PRETREND_WINDOW = (-1440, -60)
PRETREND_TOL = 0.1
WINDOW_48H = pd.Timedelta(hours=48)


# --------------------------------------------------
# Dashboard defaults
# --------------------------------------------------
DEFAULT_PRIMARY = [NH3, H2S, TEMP_NH3, TEMP_H2S, FLOW]
