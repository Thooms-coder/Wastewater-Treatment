# paths.py

from pathlib import Path
from config import PROJECT_ROOT

# --------------------------------------------------
# Data directories
# --------------------------------------------------
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# --------------------------------------------------
# Output directories
# --------------------------------------------------
PLOTS_DIR = PROJECT_ROOT / "plots"

EVENT_STUDY_PLOTS_DIR = PLOTS_DIR / "event_study"
EVENT_STUDY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)