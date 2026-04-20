import unittest

import pandas as pd

from app.data_services import (
    build_events_table,
    build_hourly_table,
    build_period_summaries,
    compute_event_metrics_table,
    compute_event_study_summary,
    enrich_operational_features,
    filter_time_indexed_df,
)
from scripts.constants import H2S, NH3


class DataServicesTests(unittest.TestCase):
    def test_build_events_table_sorts_and_labels_transitions(self):
        index = pd.date_range("2026-01-01 00:00", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "ferric_available": [0, 1, 1, 0, 0],
                "hcl_available": [0, 0, 1, 1, 0],
            },
            index=index,
        )

        result = build_events_table(df)

        self.assertEqual(list(result["chemical"]), ["Ferric", "HCl", "Ferric", "HCl"])
        self.assertEqual(list(result["event_type"]), ["ON", "ON", "OFF", "OFF"])
        self.assertEqual(list(result["timestamp"]), [index[1], index[2], index[3], index[4]])

    def test_build_period_summaries_renames_expected_columns(self):
        index = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-02-02", "2026-02-03"])
        daily = pd.DataFrame(
            {
                NH3: [10.0, 14.0, 20.0, 24.0],
                H2S: [1.0, 3.0, 5.0, 7.0],
                "total_gpm": [100.0, 120.0, 140.0, 160.0],
                "transferred_lbs_vol_daily": [50.0, 60.0, 70.0, 80.0],
            },
            index=index,
        )

        monthly, weekday = build_period_summaries(daily)

        self.assertIn("nh3_monthly_mean", monthly.columns)
        self.assertIn("h2s_monthly_mean", monthly.columns)
        self.assertIn("total_gpm_monthly_mean", monthly.columns)
        self.assertIn("transferred_lbs_vol_monthly_mean", monthly.columns)
        self.assertEqual(monthly.loc[1, "nh3_monthly_mean"], 12.0)
        self.assertEqual(monthly.loc[2, "h2s_monthly_mean"], 6.0)
        self.assertEqual(monthly.loc[1, "days_in_data"], 2)

        self.assertIn("nh3_weekday_mean", weekday.columns)
        self.assertIn("weekday_name", weekday.columns)
        self.assertTrue({"Monday", "Tuesday"}.issubset(set(weekday["weekday_name"])))

    def test_build_period_summaries_returns_none_for_empty_input(self):
        monthly, weekday = build_period_summaries(pd.DataFrame())

        self.assertIsNone(monthly)
        self.assertIsNone(weekday)

    def test_filter_time_indexed_df_slices_datetime_index(self):
        index = pd.date_range("2026-01-01 00:00", periods=6, freq="h")
        df = pd.DataFrame({"value": range(6)}, index=index)

        result = filter_time_indexed_df(df, index[1], index[3])

        self.assertEqual(list(result.index), list(index[1:4]))
        self.assertEqual(list(result["value"]), [1, 2, 3])

    def test_build_hourly_table_rolls_up_flow_and_signals(self):
        index = pd.date_range("2026-01-01 00:00", periods=4, freq="30min")
        df = pd.DataFrame(
            {
                "west_sludge_out_gpm": [10, 10, 20, 20],
                "east_sludge_out_gpm": [1, 1, 2, 2],
                "digesters_sludge_out_flow": [0, 0, 3, 3],
                NH3: [4.0, 6.0, 10.0, 12.0],
                H2S: [1.0, 3.0, 2.0, 5.0],
            },
            index=index,
        )

        hourly = build_hourly_table(df)

        self.assertIn("flow_gal_hr", hourly.columns)
        self.assertIn("lbs_volatile", hourly.columns)
        self.assertEqual(hourly.iloc[0][NH3], 4.0)
        self.assertEqual(hourly.iloc[1][NH3], 8.0)
        self.assertEqual(hourly.iloc[1][H2S], 3.0)
        self.assertGreater(hourly.iloc[0]["flow_gal_hr"], 0)

    def test_enrich_operational_features_backfills_missing_hcl_dose_columns(self):
        index = pd.date_range("2025-10-02 14:30", periods=3, freq="h")
        df = pd.DataFrame(
            {
                "west_sludge_out_gpm": [100.0, 100.0, 100.0],
                "east_sludge_out_gpm": [50.0, 50.0, 50.0],
                "digesters_sludge_out_flow": [25.0, 25.0, 25.0],
                "hcl_available": [1, 1, 0],
                "ferric_available": [1, 1, 1],
            },
            index=index,
        )

        result = enrich_operational_features(df)

        self.assertIn("hcl_solution_lbs_per_day", result.columns)
        self.assertIn("hcl_active_lbs_per_day", result.columns)
        self.assertGreater(result.iloc[0]["hcl_active_lbs_per_day"], 0)
        self.assertEqual(result.iloc[2]["hcl_active_lbs_per_day"], 0)

    def test_compute_event_metrics_table_computes_pre_post_effects(self):
        index = pd.date_range("2026-01-01 00:00", periods=220, freq="h")
        ferric = [0] * len(index)
        ferric[72:] = [1] * (len(index) - 72)

        nh3 = []
        h2s = []
        for i in range(len(index)):
            if i < 72 - 12:
                nh3.append(10.0)
                h2s.append(4.0)
            elif i <= 72 + 12:
                nh3.append(9.0)
                h2s.append(3.5)
            else:
                nh3.append(5.0)
                h2s.append(2.0)

        df = pd.DataFrame(
            {
                "ferric_available": ferric,
                "hcl_available": [0] * len(index),
                NH3: nh3,
                H2S: h2s,
            },
            index=index,
        )

        result = compute_event_metrics_table(df)
        ferric_on_nh3 = result[
            (result["chemical"] == "Ferric")
            & (result["event_type"] == "ON")
            & (result["signal"] == "NH3")
        ].iloc[0]

        self.assertEqual(ferric_on_nh3["baseline"], 10.0)
        self.assertEqual(ferric_on_nh3["post"], 5.0)
        self.assertEqual(ferric_on_nh3["delta"], -5.0)
        self.assertEqual(ferric_on_nh3["n_events"], 1)

    def test_compute_event_study_summary_returns_aligned_windows(self):
        index = pd.date_range("2026-01-01 00:00", periods=240, freq="h")
        ferric = [0] * len(index)
        ferric[80:] = [1] * (len(index) - 80)
        signal = [10.0 if i < 92 else 7.0 for i in range(len(index))]
        df = pd.DataFrame(
            {
                "ferric_available": ferric,
                "hcl_available": [0] * len(index),
                NH3: signal,
            },
            index=index,
        )

        summary, aligned_df, pretrend_ok = compute_event_study_summary(df, "Ferric", "ON", NH3)

        self.assertFalse(summary.empty)
        self.assertEqual(aligned_df.shape[1], 1)
        self.assertIn(0, summary.index)
        self.assertTrue(pretrend_ok)


if __name__ == "__main__":
    unittest.main()
