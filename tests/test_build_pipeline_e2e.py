import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import build_aggregates, build_daily
from scripts.constants import H2S, NH3


class BuildPipelineE2ETests(unittest.TestCase):
    def test_daily_and_aggregate_builds_produce_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir)
            master_path = processed_dir / "master_1min.parquet"
            daily_path = processed_dir / "master_daily.parquet"
            monthly_path = processed_dir / "monthly_summary.parquet"
            weekday_path = processed_dir / "weekday_summary.parquet"
            legacy_bundle_path = processed_dir / "aggregates.parquet"

            minute_index = pd.date_range("2026-01-01 00:00", periods=2880, freq="min")
            master_df = pd.DataFrame(
                {
                    NH3: [10.0] * 1440 + [20.0] * 1440,
                    H2S: [1.0] * 1440 + [3.0] * 1440,
                    "west_sludge_out_gpm": [10.0] * 2880,
                    "east_sludge_out_gpm": [20.0] * 2880,
                    "digesters_sludge_out_flow": [5.0] * 2880,
                    "ferric_available": [0] * 1440 + [1] * 1440,
                    "hcl_available": [1] * 2880,
                    "interp_flag": [0] * 2880,
                    "ferric_solution_lbs_per_day": [100.0] * 2880,
                    "ferric_active_lbs_per_day": [40.0] * 2880,
                    "hcl_solution_lbs_per_day": [50.0] * 2880,
                    "hcl_active_lbs_per_day": [16.0] * 2880,
                },
                index=minute_index,
            )
            master_df.to_parquet(master_path)

            with patch.object(build_daily, "MASTER_PATH", master_path), \
                 patch.object(build_daily, "OUTPUT_PATH", daily_path):
                build_daily.run_daily_aggregation()

            self.assertTrue(daily_path.exists())
            daily_df = pd.read_parquet(daily_path)
            self.assertEqual(len(daily_df), 2)
            self.assertEqual(daily_df.iloc[0][NH3], 10.0)
            self.assertEqual(daily_df.iloc[1][NH3], 20.0)
            self.assertEqual(daily_df.iloc[0][H2S], 1.0)
            self.assertEqual(daily_df.iloc[1][H2S], 3.0)
            self.assertEqual(daily_df.iloc[0]["nh3_coverage"], 1.0)
            self.assertEqual(daily_df.iloc[1]["h2s_coverage"], 1.0)
            self.assertIn("transferred_lbs_vol_daily", daily_df.columns)
            self.assertIn("total_gpm", daily_df.columns)

            with patch.object(build_aggregates, "DAILY_PATH", daily_path), \
                 patch.object(build_aggregates, "MONTHLY_PATH", monthly_path), \
                 patch.object(build_aggregates, "WEEKDAY_PATH", weekday_path), \
                 patch.object(build_aggregates, "LEGACY_BUNDLE_PATH", legacy_bundle_path):
                build_aggregates.run_aggregations()

            self.assertTrue(monthly_path.exists())
            self.assertTrue(weekday_path.exists())
            self.assertTrue(legacy_bundle_path.exists())

            monthly_df = pd.read_parquet(monthly_path)
            weekday_df = pd.read_parquet(weekday_path)
            legacy_bundle = pd.read_pickle(legacy_bundle_path)

            self.assertIn("nh3_monthly_mean", monthly_df.columns)
            self.assertIn("h2s_monthly_mean", monthly_df.columns)
            self.assertIn("total_gpm_monthly_mean", monthly_df.columns)
            self.assertIn("transferred_lbs_vol_monthly_mean", monthly_df.columns)
            self.assertEqual(monthly_df.iloc[0]["nh3_monthly_mean"], 15.0)
            self.assertEqual(monthly_df.iloc[0]["days_in_data"], 2)

            self.assertIn("weekday_name", weekday_df.columns)
            self.assertEqual(weekday_df["days_in_data"].sum(), 2)
            self.assertIn("monthly", legacy_bundle)
            self.assertIn("weekday", legacy_bundle)


if __name__ == "__main__":
    unittest.main()
