import unittest

import pandas as pd

from scripts import daily_reports


class DailyReportsTests(unittest.TestCase):
    def test_load_daily_reports_parses_dates_and_key_fields(self):
        chemical_path = daily_reports._latest_report_file(daily_reports.CHEMICAL_REPORT_GLOB)
        biosolids_path = daily_reports._latest_report_file(daily_reports.BIOSOLIDS_REPORT_GLOB)
        if chemical_path is None or biosolids_path is None:
            self.skipTest("Daily report Excel files are not present")

        chemical, _chemical_meta = daily_reports.load_daily_report(chemical_path, "chem")
        biosolids, _biosolids_meta = daily_reports.load_daily_report(biosolids_path, "bio")

        self.assertEqual(chemical.index.min(), pd.Timestamp("2025-03-01"))
        self.assertEqual(chemical.index.max(), pd.Timestamp("2026-03-01"))
        self.assertEqual(biosolids.index.min(), pd.Timestamp("2025-03-01"))
        self.assertEqual(biosolids.index.max(), pd.Timestamp("2026-03-01"))

        self.assertIn("chem_hydrochloric_acid_delivered_lbs", chemical.columns)
        self.assertGreater(chemical["chem_hydrochloric_acid_delivered_lbs"].notna().sum(), 0)
        self.assertIn("bio_centrate_ph_su", biosolids.columns)
        self.assertGreater(biosolids["bio_centrate_ph_su"].notna().sum(), 0)

    def test_build_struvite_observations_maps_yes_no_codes(self):
        biosolids_path = daily_reports._latest_report_file(daily_reports.BIOSOLIDS_REPORT_GLOB)
        if biosolids_path is None:
            self.skipTest("Biosolids daily report Excel file is not present")

        biosolids, _meta = daily_reports.load_daily_report(biosolids_path, "bio")
        observations = daily_reports.build_struvite_observations(biosolids)

        self.assertFalse(observations.empty)
        self.assertIn("struvite_observed", observations.columns)
        self.assertTrue(set(observations["observation_code"].dropna().unique()).issubset({1650.0, 3500.0}))


if __name__ == "__main__":
    unittest.main()
