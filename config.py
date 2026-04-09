# config.py

from pathlib import Path

# --------------------------------------------------
# Project paths (robust to script location)
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
(PLOTS_DIR / "explore").mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Time / preprocessing configuration
# --------------------------------------------------
TIME_COL = "timestamp"

SMOOTHING_WINDOW = 3            # number of samples, or None
RESAMPLE_RULE = "1min"          # matches water reclamation resolution
MAX_INTERP_GAP_MINUTES = 10     # conservative, defensible

LAG_MINUTES = [5, 15, 30, 60]
ROLLING_WINDOWS_MINUTES = [5, 15, 30, 60]
