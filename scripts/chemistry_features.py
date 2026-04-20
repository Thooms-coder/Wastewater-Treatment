# chemistry_features.py
from __future__ import annotations
import sys
from pathlib import Path

from config import PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT))

"""
chemistry_features.py

Adds chemistry-informed features to your time-indexed dataframe.

This module supports two chemistry inputs:
1) Fixed water chemistry snapshot (your screenshot):
   - pH fixed at 7.0
   - Total concentrations (mol/L) for selected ions
2) Operational dosing notes (your text):
   - Ferric chloride: 37.9% strength, SG=1.404
   - Example: 583 lb/day of 37.9% solution ≈ 220.9 lb/day active (100% equivalent)

Outputs (columns added):
- pH_fixed
- fixed_ion_<species>_M for each species
- ionic_strength_M (I = 0.5 * Σ c_i z_i^2)
- fixed_total_cations_M, fixed_total_anions_M
- fixed_charge_balance_eq_per_L, fixed_charge_balance_ratio
- ferric_solution_lbs_per_day, ferric_active_lbs_per_day (optional, gated by events)
- helper conversion utilities: mg/L <-> lb/day using MGD * 8.34
"""

# IMPORTANT INTERPRETATION NOTE:
# ------------------------------
# Fixed chemistry features included in this module are intended for
# contextual interpretation and descriptive covariate analysis only.
# They do NOT represent equilibrium speciation or mechanistic modeling
# of the biosolids system.


import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Fixed chemistry snapshot from your screenshot (Total C (M))
# -------------------------------------------------------------------
FIXED_PH = 7.0

# Total concentrations in mol/L (M)
FIXED_IONS_M = {
    "ca_2+": 1.760e+01,
    "cl_-":  1.900e-02,
    "cu_2+": 5.000e-03,
    "cu_1+": 5.000e-03,
    "fe_2+": 4.932e-04,
    "fe_3+": 4.932e-40,   # effectively 0
    "k_1+":  5.000e-02,
    "mg_2+": 2.700e-01,
    "nh4_1+": 3.800e-02,
}

# Charges for ionic strength / charge balance
ION_CHARGE = {
    "ca_2+": +2,
    "cl_-":  -1,
    "cu_2+": +2,
    "cu_1+": +1,
    "fe_2+": +2,
    "fe_3+": +3,
    "k_1+":  +1,
    "mg_2+": +2,
    "nh4_1+": +1,
}

# -------------------------------------------------------------------
# Ferric chloride dosing notes (optional features)
# -------------------------------------------------------------------
FERRIC_STRENGTH_FRAC = 0.379
FERRIC_SPECIFIC_GRAVITY = 1.404
FERRIC_REDUCTION_DATE = pd.Timestamp("2026-01-07")

# If you want a simple “representative feed rate” feature when ferric is available:
DEFAULT_FERRIC_SOLUTION_LB_PER_DAY = 583.0  # from your note/example

# -------------------------------------------------------------------
# Hydrochloric acid dosing notes (optional features)
# -------------------------------------------------------------------
HCL_STRENGTH_FRAC = 0.32
HCL_SPECIFIC_GRAVITY = 1.16
DEFAULT_HCL_SOLUTION_LB_PER_DAY = 6230.0  # from operations note/example


# -------------------------------------------------------------------
# Unit conversion helpers (common wastewater convention)
# -------------------------------------------------------------------
def lbs_per_day_from_mgL(mg_per_L: float, mgd: float) -> float:
    """
    Convert mg/L at flow (MGD) to lb/day (100% active basis).
    Rule of thumb: lb/day = MGD * 8.34 * mg/L
    """
    return float(mgd) * 8.34 * float(mg_per_L)


def mgL_from_lbs_per_day(lbs_per_day: float, mgd: float) -> float:
    """
    Convert lb/day (100% active) at flow (MGD) to mg/L.
    mg/L = lb/day / (MGD * 8.34)
    """
    denom = float(mgd) * 8.34
    return float(lbs_per_day) / denom if denom != 0 else np.nan


def active_lbs_from_solution_lbs(solution_lbs_per_day: float, strength_frac: float) -> float:
    """Convert solution lb/day to active (100%) lb/day using strength fraction."""
    return float(solution_lbs_per_day) * float(strength_frac)


# -------------------------------------------------------------------
# Chemistry feature builders
# -------------------------------------------------------------------
def compute_ionic_strength(ions_M: dict[str, float], charges: dict[str, int]) -> float:
    """
    Ionic strength (mol/L):
      I = 0.5 * Σ c_i * z_i^2
    """
    s = 0.0
    for sp, c in ions_M.items():
        z = charges.get(sp, 0)
        s += float(c) * (float(z) ** 2)
    return 0.5 * s


def compute_charge_balance_eq_per_L(ions_M: dict[str, float], charges: dict[str, int]) -> float:
    """
    Net charge balance (eq/L):
      Σ c_i z_i
    Ideally near 0 for a fully-specified ionic set; in practice this indicates
    "incomplete chemistry" (missing alkalinity, sulfate, etc.) or measurement mismatch.
    """
    s = 0.0
    for sp, c in ions_M.items():
        z = charges.get(sp, 0)
        s += float(c) * float(z)
    return s


def add_fixed_chemistry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds fixed chemistry snapshot features (constants) + derived ionic features.
    """
    out = df.copy()

    # Fixed pH
    out["pH_fixed"] = FIXED_PH

    # Add each fixed ion concentration as a column (M)
    for sp, c in FIXED_IONS_M.items():
        col = f"fixed_ion_{sp}_M"
        out[col] = float(c)

    # Derived ionic features
    I = compute_ionic_strength(FIXED_IONS_M, ION_CHARGE)
    out["ionic_strength_M"] = float(I)

    # Cation/anion totals (molar) from the provided ions only (not full water chemistry)
    cations = 0.0
    anions = 0.0
    for sp, c in FIXED_IONS_M.items():
        z = ION_CHARGE.get(sp, 0)
        if z > 0:
            cations += float(c)
        elif z < 0:
            anions += float(c)

    out["fixed_total_cations_M"] = float(cations)
    out["fixed_total_anions_M"] = float(anions)

    # Charge balance diagnostics (based only on the included species)
    net_eq = compute_charge_balance_eq_per_L(FIXED_IONS_M, ION_CHARGE)
    out["fixed_charge_balance_eq_per_L"] = float(net_eq)

    # Ratio form (avoid divide-by-zero)
    denom = (abs(cations) + abs(anions))
    out["fixed_charge_balance_ratio"] = float(net_eq) / denom if denom != 0 else np.nan

    return out


def add_ferric_dose_features(
    df: pd.DataFrame,
    ferric_available_col: str = "ferric_available",
    solution_lbs_per_day: float = DEFAULT_FERRIC_SOLUTION_LB_PER_DAY,
    strength_frac: float = FERRIC_STRENGTH_FRAC,
) -> pd.DataFrame:
    """
    Adds ferric dosing features based on availability and plant notes.

    Plant notes:
    - Normal dosing = 583 lb/day solution
    - After Jan 7 2026 feed rate reduced to half
    """

    out = df.copy()

    if ferric_available_col not in out.columns:
        out["ferric_solution_lbs_per_day"] = np.nan
        out["ferric_active_lbs_per_day"] = np.nan
        return out

    avail = out[ferric_available_col].fillna(0).astype(float).clip(0, 1)

    # Base full-rate solution dosing
    solution = avail * float(solution_lbs_per_day)

    # Apply ferric reduction after plant change
    reduction_mask = out.index >= FERRIC_REDUCTION_DATE
    solution.loc[reduction_mask] = solution.loc[reduction_mask] * 0.5

    out["ferric_solution_lbs_per_day"] = solution

    # Active ferric equivalent
    out["ferric_active_lbs_per_day"] = (
        out["ferric_solution_lbs_per_day"] * float(strength_frac)
    )

    out["ferric_strength_frac"] = float(strength_frac)
    out["ferric_specific_gravity"] = float(FERRIC_SPECIFIC_GRAVITY)

    return out


def add_hcl_dose_features(
    df: pd.DataFrame,
    hcl_available_col: str = "hcl_available",
    solution_lbs_per_day: float = DEFAULT_HCL_SOLUTION_LB_PER_DAY,
    strength_frac: float = HCL_STRENGTH_FRAC,
) -> pd.DataFrame:
    """
    Add simple HCl dosing features from availability flags and a representative feed rate.

    Current logic:
    - If HCl is available, assume representative solution dosing is active.
    - If not available, dose is 0.

    This is intentionally a descriptive feature and should be replaced with measured HCl
    feed data when those data become available.
    """
    out = df.copy()

    if hcl_available_col not in out.columns:
        out["hcl_solution_lbs_per_day"] = np.nan
        out["hcl_active_lbs_per_day"] = np.nan
        return out

    avail = out[hcl_available_col].fillna(0).astype(float).clip(0, 1)
    solution = avail * float(solution_lbs_per_day)

    out["hcl_solution_lbs_per_day"] = solution
    out["hcl_active_lbs_per_day"] = (
        out["hcl_solution_lbs_per_day"] * float(strength_frac)
    )
    out["hcl_strength_frac"] = float(strength_frac)
    out["hcl_specific_gravity"] = float(HCL_SPECIFIC_GRAVITY)

    return out


def build_chemistry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: add fixed chemistry + operational dosing features (if possible).
    """
    out = add_fixed_chemistry_features(df)
    out = add_ferric_dose_features(out)
    out = add_hcl_dose_features(out)
    return out


# -------------------------------------------------------------------
# Script entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Typical pipeline usage:
    # load -> preprocess -> features -> events -> chemistry_features
    from scripts.load_data import load_all_data
    from scripts.preprocess import preprocess_data
    from scripts.features import build_features
    from scripts.events import add_event_flags

    raw = load_all_data()
    clean = preprocess_data(raw)

    feat_df, _targets, _derived = build_features(clean)
    # Add event flags first (events operate on the dataframe)
    with_events = add_event_flags(feat_df)

    # Then add chemistry-informed features
    chem_df = build_chemistry_features(with_events)

    # Minimal sanity print
    cols_preview = [
        "pH_fixed",
        "ionic_strength_M",
        "fixed_charge_balance_eq_per_L",
        "fixed_charge_balance_ratio",
        "ferric_available",
        "ferric_solution_lbs_per_day",
        "ferric_active_lbs_per_day",
        "hcl_available",
        "hcl_solution_lbs_per_day",
        "hcl_active_lbs_per_day",
    ]
    cols_preview = [c for c in cols_preview if c in chem_df.columns]

    print(chem_df[cols_preview].tail(10))
    print("\nChemistry feature columns added:")
    added = [c for c in chem_df.columns if c.startswith("fixed_") or c in {
        "pH_fixed", "ionic_strength_M", "ferric_solution_lbs_per_day", "ferric_active_lbs_per_day",
        "hcl_solution_lbs_per_day", "hcl_active_lbs_per_day",
        "ferric_strength_frac", "ferric_specific_gravity",
        "hcl_strength_frac", "hcl_specific_gravity",
        "fixed_total_cations_M", "fixed_total_anions_M",
        "fixed_charge_balance_eq_per_L", "fixed_charge_balance_ratio"
    }]
    print(sorted(set(added)))
