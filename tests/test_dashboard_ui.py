import math
import unittest

import pandas as pd

from app.dashboard_ui import (
    build_chemistry_review_table,
    build_methods_log_template_df,
    build_research_progress_df,
    build_thesis_outline_df,
    compute_ferric_mgL_series,
    compute_hcl_mgL_series,
    normalize_optional_table,
    validate_optional_table_schema,
)


class DashboardUiTests(unittest.TestCase):
    def test_compute_ferric_mgl_series_fallback_includes_specific_gravity(self):
        # No reported value -> fallback uses the plant equation incl. the 1.404
        # ferric specific-gravity term: mg/L = lbs / (MGD * 8.34 * 1.404).
        index = pd.date_range("2026-01-01", periods=2, freq="D")
        df = pd.DataFrame(
            {
                "ferric_active_lbs_per_day": [8.34, 16.68],
                "total_gpm": [694.4444444444, 694.4444444444],
            },
            index=index,
        )

        result = compute_ferric_mgL_series(df)

        self.assertAlmostEqual(result.iloc[0], 1.0 / 1.404, places=6)
        self.assertAlmostEqual(result.iloc[1], 2.0 / 1.404, places=6)
        self.assertEqual(result.name, "ferric_active_mg_per_L")

    def test_compute_ferric_mgl_series_prefers_plant_reported_value(self):
        index = pd.date_range("2026-01-01", periods=2, freq="D")
        df = pd.DataFrame(
            {
                "chem_ferric_chloride_totes_applied_at_surge_tank_mg_l": [74.0, 120.0],
                "ferric_active_lbs_per_day": [220.0, 220.0],
                "total_gpm": [64.0, 64.0],  # would give a bogus spike if computed
            },
            index=index,
        )

        result = compute_ferric_mgL_series(df)

        self.assertAlmostEqual(result.iloc[0], 74.0, places=6)
        self.assertAlmostEqual(result.iloc[1], 120.0, places=6)

    def test_compute_hcl_mgl_series_fallback_includes_specific_gravity(self):
        # No reported dosage -> fallback uses the plant equation incl. the 1.16
        # HCl specific-gravity term: mg/L = lbs / (MGD * 8.34 * 1.16).
        index = pd.date_range("2026-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "hcl_active_lbs_per_day": [8.34, 8.34, None],
                "total_gpm": [694.4444444444, 0.0, 694.4444444444],
            },
            index=index,
        )

        result = compute_hcl_mgL_series(df)

        self.assertAlmostEqual(result.iloc[0], 1.0 / 1.16, places=6)
        self.assertTrue(math.isnan(result.iloc[1]))
        self.assertTrue(math.isnan(result.iloc[2]))

    def test_compute_hcl_mgl_series_prefers_plant_reported_value(self):
        # When the plant-reported HCl dosage is present it is authoritative and
        # is used verbatim (not recomputed from flow).
        index = pd.date_range("2026-01-01", periods=2, freq="D")
        df = pd.DataFrame(
            {
                "hcl_active_mg_per_L_reported": [360.0, 452.0],
                "hcl_active_lbs_per_day": [6180.0, 6180.0],
                "total_gpm": [64.0, 64.0],  # would give a bogus ~2000 if computed
            },
            index=index,
        )

        result = compute_hcl_mgL_series(df)

        self.assertAlmostEqual(result.iloc[0], 360.0, places=6)
        self.assertAlmostEqual(result.iloc[1], 452.0, places=6)

    def test_build_chemistry_review_table_reports_available_hcl_features(self):
        df = pd.DataFrame(
            {
                "ferric_solution_lbs_per_day": [100.0, 120.0],
                "ferric_active_lbs_per_day": [40.0, 48.0],
                "hcl_solution_lbs_per_day": [50.0, 60.0],
                "hcl_active_lbs_per_day": [16.0, 19.2],
                "total_gpm": [800.0, 900.0],
                "hcl_active_mg_per_L": [2.1, 2.4],
            }
        )

        result = build_chemistry_review_table(df)
        hcl_features_row = result[result["workflow_element"] == "HCl dosing features"].iloc[0]

        self.assertEqual(hcl_features_row["status"], "Available")
        self.assertIn("hcl_solution_lbs_per_day", hcl_features_row["current_value"])
        self.assertIn("hcl_active_lbs_per_day", hcl_features_row["current_value"])

    def test_normalize_optional_table_parses_known_date_columns(self):
        df = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02"],
                "sample_time": ["2026-01-03 10:00", "bad-date"],
                "value": [1, 2],
            }
        )

        result = normalize_optional_table(df)

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["date"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["sample_time"]))
        self.assertTrue(pd.isna(result.loc[1, "sample_time"]))

    def test_validate_optional_table_schema_flags_missing_required_groups(self):
        df = pd.DataFrame({"notes": ["only notes"]})

        result = validate_optional_table_schema(df, "Struvite Observations")

        self.assertFalse(result["is_valid"])
        self.assertEqual(len(result["missing_required_groups"]), 2)

    def test_validate_optional_table_schema_accepts_minimum_required_groups(self):
        df = pd.DataFrame(
            {
                "date": ["2026-01-01"],
                "location": ["centrate line"],
            }
        )

        result = validate_optional_table_schema(df, "Struvite Observations")

        self.assertTrue(result["is_valid"])
        self.assertIn("severity", result["missing_recommended"])

    def test_research_progress_and_outline_helpers_return_expected_sections(self):
        progress = build_research_progress_df()
        methods_template = build_methods_log_template_df()
        outline = build_thesis_outline_df()

        self.assertIn("Full-scale odor analytics", set(progress["lane"]))
        self.assertIn("Bench-scale H2S method", set(progress["lane"]))
        self.assertIn("experiment_or_method", methods_template.columns)
        self.assertIn("Full-scale analytics", set(methods_template["lane"]))
        self.assertIn("1. Introduction and problem framing", set(outline["section"]))
        self.assertIn("5. Full-scale Dayton analysis", set(outline["section"]))


if __name__ == "__main__":
    unittest.main()
