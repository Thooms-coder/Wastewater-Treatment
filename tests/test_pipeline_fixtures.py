import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.data_services import load_all_frames
from scripts.constants import H2S, NH3


class PipelineFixtureTests(unittest.TestCase):
    def setUp(self):
        load_all_frames.clear()

    def tearDown(self):
        load_all_frames.clear()

    def test_load_all_frames_reads_processed_fixture_and_builds_hourly_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            master_path = root / "master_1min.parquet"
            hourly_path = root / "missing_hourly.parquet"
            daily_path = root / "master_daily.parquet"
            monthly_path = root / "monthly_summary.parquet"
            weekday_path = root / "weekday_summary.parquet"
            event_metrics_path = root / "event_metrics.csv"
            struvite_path = root / "struvite_observations.csv"
            labs_path = root / "chemistry_lab_results.csv"

            minute_index = pd.date_range("2026-01-01 00:00", periods=4, freq="30min")
            master_df = pd.DataFrame(
                {
                    "west_sludge_out_gpm": [10, 10, 20, 20],
                    "east_sludge_out_gpm": [1, 1, 2, 2],
                    "digesters_sludge_out_flow": [0, 0, 3, 3],
                    NH3: [4.0, 6.0, 10.0, 12.0],
                    H2S: [1.0, 3.0, 2.0, 5.0],
                },
                index=minute_index,
            )
            daily_df = pd.DataFrame(
                {NH3: [5.0], H2S: [2.0]},
                index=pd.to_datetime(["2026-01-01"]),
            )
            monthly_df = pd.DataFrame({"nh3_monthly_mean": [5.0]}, index=pd.Index([1]))
            weekday_df = pd.DataFrame({"nh3_weekday_mean": [5.0]}, index=pd.Index([2]))
            event_metrics_df = pd.DataFrame({"chemical": ["Ferric"]})
            struvite_df = pd.DataFrame({"date": ["2026-01-01"], "location": ["line-a"]})
            labs_df = pd.DataFrame({"date": ["2026-01-01"], "pH": [7.0]})

            master_df.to_parquet(master_path)
            daily_df.to_parquet(daily_path)
            monthly_df.to_parquet(monthly_path)
            weekday_df.to_parquet(weekday_path)
            event_metrics_df.to_csv(event_metrics_path, index=False)
            struvite_df.to_csv(struvite_path, index=False)
            labs_df.to_csv(labs_path, index=False)

            master, hourly, daily, monthly, weekday, event_metrics, struvite_obs, chem_labs = load_all_frames(
                master_path,
                hourly_path,
                daily_path,
                monthly_path,
                weekday_path,
                event_metrics_path,
                struvite_path,
                labs_path,
            )

            self.assertIn("total_gpm", master.columns)
            self.assertIn("lbs_per_min", master.columns)
            self.assertIn("transferred_lbs_vol", master.columns)
            self.assertIsNotNone(hourly)
            self.assertIn("flow_gal_hr", hourly.columns)
            self.assertIsNotNone(daily)
            self.assertIsNotNone(monthly)
            self.assertIsNotNone(weekday)
            self.assertEqual(event_metrics.iloc[0]["chemical"], "Ferric")
            self.assertIn("date", struvite_obs.columns)
            self.assertIn("pH", chem_labs.columns)


if __name__ == "__main__":
    unittest.main()
