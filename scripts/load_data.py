# load_data.py

import sys
from pathlib import Path
import pandas as pd

from config import PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT))

from scripts.paths import RAW_DATA_DIR


# --------------------------------------------------
# Plant local timezone
# --------------------------------------------------
# Gas-sensor exports store an authoritative UTC "ISO time" column plus a naive
# local "Time Stamp". The per-session "Time Zone Offset" header is inconsistent
# and does not reliably track DST, so we anchor on the UTC column and convert to
# true plant-local (Eastern, DST-aware). Water Reclamation SCADA exports are
# already recorded in plant-local wall-clock time, so both sources end up on the
# same naive local timeline.
PLANT_TZ = "America/New_York"


# --------------------------------------------------
# Utility: normalize column names
# --------------------------------------------------
def normalize_columns(df):
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    return df


# --------------------------------------------------
# Utility: safe datetime parsing (naive local wall-clock strings)
# --------------------------------------------------
def parse_datetime(series):
    """
    Robust datetime parsing supporting:
    - 11/10/25 12:01:00 AM  (plant export)
    - 9-12-2025, 7-22-54 AM (gas CSV)
    - 2025-11-10 00:01:00   (ISO)

    Returns tz-naive timestamps (assumed already in plant-local wall-clock time).
    """

    s = series.astype(str).str.strip()

    # 1) Plant / gas "M/D/YY(YY), h:mm:ss AM" format
    dt = pd.to_datetime(s, format="%m/%d/%Y, %I:%M:%S %p", errors="coerce")

    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(
            s[mask], format="%m/%d/%y %I:%M:%S %p", errors="coerce"
        )

    # 2) Gas CSV dash format (rare in file content, common in filenames)
    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(
            s[mask], format="%m-%d-%Y, %I-%M-%S %p", errors="coerce"
        )

    # 3) Final fallback (ISO or other valid strings)
    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(s[mask], errors="coerce")

    return dt


def parse_iso_utc_to_local(series):
    """
    Parse the authoritative UTC "ISO time" column (e.g. 2025-09-12T11:22:54.000Z)
    and convert to tz-naive plant-local wall-clock time (DST-aware).
    """
    dt = pd.to_datetime(series.astype(str).str.strip(), utc=True, errors="coerce")
    local = dt.dt.tz_convert(PLANT_TZ).dt.tz_localize(None)
    return local


# --------------------------------------------------
# Utility: remove duplicate columns safely
# --------------------------------------------------
def remove_duplicate_columns(df, label=""):
    dups = df.columns[df.columns.duplicated()].tolist()
    if dups:
        print(f"[WARN] Dropping duplicate columns in {label}: {dups}")
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


# --------------------------------------------------
# Gas-file header detection (robust to variable metadata blocks)
# --------------------------------------------------
def _find_gas_header_line_csv(path):
    """
    Return the 0-indexed line where the "Time Stamp" header sits in a gas CSV.
    Gas exports carry a variable-length metadata block (7-11 rows depending on
    Name/Session 2nd Title/operator notes), so the header must be detected per
    file rather than hardcoded to skiprows=8.
    """
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if line.strip().lower().startswith("time stamp"):
                return i
            if i > 40:
                break
    return None


def _find_gas_header_row_excel(raw):
    """Row index whose first cell is the 'Time Stamp' label (Excel gas export)."""
    for i in range(min(len(raw), 30)):
        first = str(raw.iloc[i, 0]).strip().lower()
        if first.startswith("time stamp"):
            return i
    return None


def _read_gas_file(path, prefix):
    """
    Load a single H2S/NH3 export (.csv or .xlsx) into a time-indexed frame.

    - Detects the "Time Stamp" header row dynamically.
    - Anchors the index on the UTC ISO column (converted to plant-local),
      falling back to the naive local "Time Stamp" where ISO is missing.
    - Normalizes and prefixes the value columns (h2s_ / nh3_).
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        header_line = _find_gas_header_line_csv(path)
        if header_line is None:
            print(f"[WARN] No 'Time Stamp' header found in {path.name}; skipping.")
            return None
        df = pd.read_csv(
            path, skiprows=header_line, engine="python", encoding="latin-1"
        )
    else:
        raw = pd.read_excel(path, sheet_name=0, header=None, dtype=str)
        header_row = _find_gas_header_row_excel(raw)
        if header_row is None:
            print(f"[WARN] No 'Time Stamp' header found in {path.name}; skipping.")
            return None
        df = pd.read_excel(path, sheet_name=0, header=header_row)

    df = df.dropna(how="all")
    df = normalize_columns(df)

    # Drop trailing empty/unnamed columns produced by ragged export rows.
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]

    # --------------------------------------------------
    # Build a plant-local timestamp index
    # --------------------------------------------------
    ts = None
    if "iso_time" in df.columns:
        ts = parse_iso_utc_to_local(df["iso_time"])

    if "time_stamp" in df.columns:
        local = parse_datetime(df["time_stamp"])
        ts = local if ts is None else ts.fillna(local)

    if ts is None:
        print(f"[WARN] No time columns in {path.name}; skipping.")
        return None

    df = df.assign(_ts=ts.values).dropna(subset=["_ts"]).set_index("_ts")
    df.index.name = "timestamp"

    if df.empty:
        print(f"[WARN] All timestamps invalid in {path.name}; skipping.")
        return None

    # Drop the raw time columns; they are now the index / redundant.
    df = df.drop(columns=[c for c in ("time_stamp", "iso_time") if c in df.columns])

    # Coerce values to numeric and prefix.
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.add_prefix(prefix)

    return df


def _load_gas_stream(token, prefix):
    """Load and merge every H2S/NH3 export (.csv and .xlsx) for one stream."""
    files = sorted(RAW_DATA_DIR.glob(f"*{token}*.csv")) + sorted(
        RAW_DATA_DIR.glob(f"*{token}*.xlsx")
    )
    if not files:
        raise ValueError(f"No {token} files found.")

    frames = [f for f in (_read_gas_file(p, prefix) for p in files) if f is not None]
    if not frames:
        raise ValueError(f"No usable {token} data could be loaded.")

    combined = pd.concat(frames).sort_index()
    combined = remove_duplicate_columns(combined, token)

    # Merge overlapping/duplicate timestamps from re-exported files by keeping
    # the first non-null value per column (rather than discarding whole rows).
    if combined.index.duplicated().any():
        combined = combined.groupby(level=0).first()

    return combined


def detect_water_header_row(raw):
    """
    Some plant exports contain a duplicated two-row header where the first row has
    a generic label like "Digesters Sludge Out Flow" and the second row contains
    the real West/East flow column names. Prefer the row with the strongest match
    to the expected operational headers.
    """
    header_candidates = []
    expected_terms = [
        "west sludge out",
        "east sludge out",
        "eest sludge out",
        "digesters sludge out flow",
        "gbt sludge feed pump",
    ]

    for i in range(min(len(raw), 12)):
        values = raw.iloc[i].astype(str).str.strip()
        if not values.str.contains("Time", case=False).any():
            continue

        lowered = values.str.lower()
        score = sum(lowered.str.contains(term, regex=False).sum() for term in expected_terms)
        unnamed_penalty = lowered.str.startswith("unnamed").sum()
        header_candidates.append((score, -unnamed_penalty, i))

    if not header_candidates:
        return None

    header_candidates.sort(reverse=True)
    return header_candidates[0][2]


# --------------------------------------------------
# H2S / NH3 Loaders
# --------------------------------------------------
def load_h2s_data():
    return _load_gas_stream("H2S", "h2s_")


def load_nh3_data():
    return _load_gas_stream("NH3", "nh3_")


# --------------------------------------------------
# Water Reclamation Loader
# --------------------------------------------------
def load_water_reclamation_data():
    files = sorted(RAW_DATA_DIR.glob("Water Reclamation*.xlsx"))
    if not files:
        raise ValueError("No water reclamation Excel files found.")

    dfs = []

    for file in files:

        # Read raw without header
        raw = pd.read_excel(file, header=None)

        # Find the strongest header row containing the real flow labels.
        header_row = detect_water_header_row(raw)

        if header_row is None:
            raise ValueError(f"Could not find header row in {file.name}")

        # Re-read using detected header row
        df = pd.read_excel(file, header=header_row)

        # Drop completely empty rows
        df = df.dropna(how="all")

        # Normalize column names
        df = normalize_columns(df)

        if "time" not in df.columns:
            raise ValueError(f"'Time' column missing after normalization in {file.name}")

        # Parse datetime (plant-local wall-clock)
        df["time"] = parse_datetime(df["time"])
        df = df.dropna(subset=["time"]).set_index("time")
        df.index.name = "timestamp"

        dfs.append(df)

    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    combined = remove_duplicate_columns(combined, "Water")

    return combined


# --------------------------------------------------
# Master Merge
# --------------------------------------------------
def load_all_data():
    h2s = load_h2s_data()
    nh3 = load_nh3_data()
    water = load_water_reclamation_data()

    df = h2s.join(nh3, how="outer")
    df = df.join(water, how="outer")

    df = df.sort_index()
    df = remove_duplicate_columns(df, "MASTER")

    # Guard against NaT index
    if df.index.isna().any():
        raise ValueError("NaT detected in master index.")

    if df.index.min() is pd.NaT or df.index.max() is pd.NaT:
        raise ValueError("Invalid index range detected.")

    return df


# --------------------------------------------------
# Debug
# --------------------------------------------------
if __name__ == "__main__":
    df = load_all_data()
    print(df.head())
    print(df.info())
