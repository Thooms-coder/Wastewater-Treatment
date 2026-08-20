from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, RAW_DATA_DIR
from scripts.paths import PROCESSED_DATA_DIR


CHEMICAL_REPORT_GLOB = "Chemical Treatment__*.xlsx"
BIOSOLIDS_REPORT_GLOB = "Biosolids Dewatering Facility__*.xlsx"

CHEMICAL_DAILY_PATH = PROCESSED_DATA_DIR / "chemical_treatment_daily.parquet"
BIOSOLIDS_DAILY_PATH = PROCESSED_DATA_DIR / "biosolids_dewatering_daily.parquet"
CHEM_LABS_CSV_PATH = PROCESSED_DATA_DIR / "chemistry_lab_results.csv"
STRUVITE_OBS_CSV_PATH = PROCESSED_DATA_DIR / "struvite_observations.csv"
REPORT_METADATA_PATH = PROCESSED_DATA_DIR / "daily_report_metadata.json"
HCL_STRENGTH_FRAC_FOR_REPORTS = 0.32

SUMMARY_LABELS = {"MINIMUM", "MAXIMUM", "AVERAGE", "SUM", "GEOMEAN"}
MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
WEEKDAYS = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


def _report_search_dirs() -> list[Path]:
    dirs = [RAW_DATA_DIR, DATA_DIR]
    seen = set()
    out = []
    for path in dirs:
        if path not in seen:
            out.append(path)
            seen.add(path)
    return out


def find_report_files(pattern: str) -> list[Path]:
    files: list[Path] = []
    for directory in _report_search_dirs():
        files.extend(directory.glob(pattern))
    return sorted(set(files), key=lambda p: (p.stat().st_mtime, p.name))


def _latest_report_file(pattern: str) -> Path | None:
    files = find_report_files(pattern)
    return files[-1] if files else None


def _extract_export_year(path: Path) -> int | None:
    match = re.search(r"__(\d{2})_(\d{2})_(\d{4})__", path.name)
    return int(match.group(3)) if match else None


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("%", "percent")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def _unit_slug(value: object) -> str:
    if pd.isna(value):
        return ""
    return _slug(value)


def _make_unique_columns(names: list[str], tags: list[object]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    for name, tag in zip(names, tags):
        base = name
        if base in counts:
            tag_text = _slug(tag)
            candidate = f"{base}_{tag_text}" if tag_text != "unnamed" else f"{base}_{counts[base] + 1}"
        else:
            candidate = base

        while candidate in counts:
            candidate = f"{base}_{counts[base] + 1}"

        counts[base] = counts.get(base, 0) + 1
        counts[candidate] = counts.get(candidate, 0)
        out.append(candidate)
    return out


def _coerce_numeric(series: pd.Series, struvite: bool = False) -> pd.Series:
    if series.dtype == object:
        cleaned = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        # The yes/no -> observation-code substitution is meaningful only for the
        # struvite observation column; applying it to every text column would
        # silently turn unrelated "yes"/"no" cells into the struvite sentinels.
        if struvite:
            cleaned = cleaned.str.lower().replace({"yes": "3500", "no": "1650"})
        cleaned = cleaned.mask(cleaned.isin(["", "nan", "NaN"]), np.nan)
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _infer_start_year(rows: pd.DataFrame, export_year: int | None) -> int:
    if export_year is None:
        export_year = pd.Timestamp.today().year

    first = rows.iloc[0]
    month_num = MONTHS[str(first.iloc[0]).strip().lower()[:3]]
    day_num, weekday_text = _parse_day_weekday(first)
    expected_weekday = WEEKDAYS.get(weekday_text)

    candidates = [export_year, export_year - 1]
    if expected_weekday is not None:
        for year in candidates:
            try:
                if pd.Timestamp(year=year, month=month_num, day=day_num).weekday() == expected_weekday:
                    return year
            except ValueError:
                continue

    return export_year - 1


def _parse_day_weekday(row: pd.Series) -> tuple[int, str]:
    day_text = str(row.iloc[1]).strip()
    day_match = re.search(r"\d+", day_text)
    if day_match is None:
        raise ValueError(f"Could not parse day from report row: {day_text!r}")
    day_num = int(day_match.group(0))

    weekday_text = ""
    if len(row) > 2 and pd.notna(row.iloc[2]):
        weekday_text = str(row.iloc[2]).strip().lower()[:3]
    else:
        parts = re.findall(r"[A-Za-z]+", day_text)
        weekday_text = parts[0].lower()[:3] if parts else ""

    return day_num, weekday_text


def _build_daily_index(body: pd.DataFrame, export_year: int | None) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    rows = body.copy()
    rows = rows[~rows.iloc[:, 0].astype(str).str.upper().isin(SUMMARY_LABELS)]
    rows = rows[rows.iloc[:, 1].notna()]
    rows.iloc[:, 0] = rows.iloc[:, 0].ffill()
    rows = rows[rows.iloc[:, 0].notna()]

    start_year = _infer_start_year(rows, export_year)
    current_year = start_year
    previous_month = None
    dates = []

    for _, row in rows.iterrows():
        month_key = str(row.iloc[0]).strip().lower()[:3]
        month_num = MONTHS.get(month_key)
        if month_num is None:
            dates.append(pd.NaT)
            continue

        if previous_month is not None and month_num < previous_month:
            current_year += 1
        previous_month = month_num

        try:
            day_num, _weekday_text = _parse_day_weekday(row)
            dates.append(pd.Timestamp(year=current_year, month=month_num, day=day_num))
        except (TypeError, ValueError):
            dates.append(pd.NaT)

    index = pd.DatetimeIndex(dates, name="date")
    keep = ~index.isna()
    return rows.loc[keep].copy(), index[keep]


def load_daily_report(path: Path, prefix: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_excel(path, sheet_name="Data", header=None)
    comments = pd.read_excel(path, sheet_name="Comments", header=None)

    names = raw.iloc[2, 3:].tolist()
    units = raw.iloc[3, 3:].tolist()
    tags = raw.iloc[1, 3:].tolist()
    areas = raw.iloc[0, 3:].tolist()

    base_columns = []
    for name, unit in zip(names, units):
        unit_part = _unit_slug(unit)
        base = f"{prefix}_{_slug(name)}"
        if unit_part:
            base = f"{base}_{unit_part}"
        base_columns.append(base)
    value_columns = _make_unique_columns(base_columns, tags)

    body, index = _build_daily_index(raw.iloc[4:].copy(), _extract_export_year(path))
    values = body.iloc[:, 3 : 3 + len(value_columns)].copy()
    values.columns = value_columns

    out = pd.DataFrame(index=index)
    for col in value_columns:
        is_struvite = "struvite" in col.lower()
        out[col] = _coerce_numeric(values[col], struvite=is_struvite).to_numpy()

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    comment_body, comment_index = _build_daily_index(comments.iloc[4:].copy(), _extract_export_year(path))
    comment_values = comment_body.iloc[:, 3 : 3 + len(value_columns)].copy()
    comment_count = int(comment_values.notna().sum().sum())

    metadata = {
        "source_file": str(path),
        "rows": int(len(out)),
        "columns": int(len(out.columns)),
        "start": str(out.index.min()) if not out.empty else None,
        "end": str(out.index.max()) if not out.empty else None,
        "comment_cells": comment_count,
        "fields": [
            {
                "column": column,
                "source_name": str(name),
                "unit": None if pd.isna(unit) else str(unit),
                "source_id": None if pd.isna(tag) else str(tag),
                "source_area": None if pd.isna(area) else str(area),
            }
            for column, name, unit, tag, area in zip(value_columns, names, units, tags, areas)
        ],
    }

    return out, metadata


def load_chemical_treatment_daily() -> tuple[pd.DataFrame | None, dict | None]:
    path = _latest_report_file(CHEMICAL_REPORT_GLOB)
    if path is None:
        return None, None
    return load_daily_report(path, "chem")


def load_biosolids_dewatering_daily() -> tuple[pd.DataFrame | None, dict | None]:
    path = _latest_report_file(BIOSOLIDS_REPORT_GLOB)
    if path is None:
        return None, None
    return load_daily_report(path, "bio")


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series(np.nan, index=df.index, dtype=float)


def build_chemistry_lab_results(biosolids: pd.DataFrame | None) -> pd.DataFrame:
    if biosolids is None or biosolids.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=biosolids.index)
    out["date"] = biosolids.index
    out["sample_location"] = "biosolids_dewatering"
    out["centrate_pH"] = _first_existing(biosolids, ["bio_centrate_ph_su"])
    out["centrate_alkalinity_mg_L"] = _first_existing(biosolids, ["bio_centrate_alkalinity_mg_l"])
    out["centrate_tss_mg_L"] = _first_existing(biosolids, ["bio_centrate_tss_mg_l"])
    out["centrate_ortho_p_mg_L"] = _first_existing(biosolids, ["bio_biosolids_centrate_ortho_p_mg_l"])
    out["centrate_total_p_mg_L"] = _first_existing(biosolids, ["bio_biosolids_centrate_total_phophorus_mg_l"])
    out["filtrate_pH"] = _first_existing(biosolids, ["bio_filtrate_ph_su"])
    out["filtrate_alkalinity_mg_L"] = _first_existing(biosolids, ["bio_filtrate_alkalinity_mg_l"])
    out["filtrate_tss_mg_L"] = _first_existing(biosolids, ["bio_filtrate_tss_mg_l"])
    out["filtrate_ortho_p_mg_L"] = _first_existing(biosolids, ["bio_biosolids_filtrate_ortho_p_mg_l"])
    out["filtrate_total_p_mg_L"] = _first_existing(biosolids, ["bio_biosolids_filtrate_total_phosphorus_mg_l"])

    value_cols = [c for c in out.columns if c not in {"date", "sample_location"}]
    return out.loc[out[value_cols].notna().any(axis=1)].reset_index(drop=True)


def build_struvite_observations(biosolids: pd.DataFrame | None) -> pd.DataFrame:
    if biosolids is None or biosolids.empty:
        return pd.DataFrame()

    col = "bio_biocake_struvite_observation_no_1650_yes_3500"
    if col not in biosolids.columns:
        return pd.DataFrame()

    observed = biosolids[col].copy()
    severity = pd.Series(np.nan, index=biosolids.index, dtype=object)
    severity.loc[observed.notna() & (observed >= 3000)] = "observed"
    severity.loc[observed.notna() & (observed < 3000)] = "not_observed"
    out = pd.DataFrame(
        {
            "date": biosolids.index,
            "location": "biocake",
            "struvite_observed": observed.where(observed.isna(), observed >= 3000),
            "observation_code": observed,
            "severity": severity,
        }
    )
    return out.loc[out["observation_code"].notna()].reset_index(drop=True)


def build_daily_report_features(
    chemical: pd.DataFrame | None,
    biosolids: pd.DataFrame | None,
) -> pd.DataFrame:
    frames = [df for df in [chemical, biosolids] if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1).sort_index()

    if chemical is not None and not chemical.empty:
        # Only accept a genuine lb/day column here. Falling back to an mg/L
        # column would silently store a concentration as a mass rate (and then
        # multiply it by the strength fraction), producing dimensionally wrong
        # "active lb/day" values.
        ferric_solution = _first_existing(
            chemical,
            [
                "chem_ferric_chloride_totes_applied_at_surge_tank_lbs_day",
            ],
        )
        out["ferric_solution_lbs_per_day_measured"] = ferric_solution

        ferric_strength = _first_existing(chemical, ["chem_ferric_chloride_strength_percent"]) / 100.0
        out["ferric_strength_frac_measured"] = ferric_strength
        out["ferric_specific_gravity_measured"] = _first_existing(
            chemical, ["chem_ferric_choride_specific_gravity_fecl3"]
        )
        out["ferric_active_lbs_per_day_measured"] = ferric_solution * ferric_strength

        hcl_solution = _first_existing(chemical, ["chem_hydrochloric_acid_delivered_lbs"])
        out["hcl_solution_lbs_per_day_measured"] = hcl_solution
        out["hcl_active_lbs_per_day_measured"] = hcl_solution * HCL_STRENGTH_FRAC_FOR_REPORTS
        out["hcl_dosage_mg_per_L_measured"] = _first_existing(chemical, ["chem_hydrochloric_acid_dosage_mg_l"])

    return out


def align_daily_report_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    chemical, _ = load_chemical_treatment_daily()
    biosolids, _ = load_biosolids_dewatering_daily()
    daily = build_daily_report_features(chemical, biosolids)
    if daily.empty:
        return pd.DataFrame(index=index)

    aligned = daily.reindex(pd.DatetimeIndex(index).normalize())
    aligned.index = index
    return aligned


def write_daily_report_outputs() -> dict:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    chemical, chemical_meta = load_chemical_treatment_daily()
    biosolids, biosolids_meta = load_biosolids_dewatering_daily()

    metadata = {}
    if chemical is not None:
        chemical.to_parquet(CHEMICAL_DAILY_PATH)
        metadata["chemical_treatment"] = chemical_meta
    if biosolids is not None:
        biosolids.to_parquet(BIOSOLIDS_DAILY_PATH)
        metadata["biosolids_dewatering"] = biosolids_meta

    chem_labs = build_chemistry_lab_results(biosolids)
    if not chem_labs.empty:
        chem_labs.to_csv(CHEM_LABS_CSV_PATH, index=False)
        metadata["chemistry_lab_results"] = {
            "rows": int(len(chem_labs)),
            "columns": list(chem_labs.columns),
        }

    struvite = build_struvite_observations(biosolids)
    if not struvite.empty:
        struvite.to_csv(STRUVITE_OBS_CSV_PATH, index=False)
        metadata["struvite_observations"] = {
            "rows": int(len(struvite)),
            "columns": list(struvite.columns),
        }

    REPORT_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    meta = write_daily_report_outputs()
    for name, info in meta.items():
        rows = info.get("rows") if isinstance(info, dict) else None
        print(f"✓ {name}: {rows if rows is not None else 'written'}")
