import unittest

import pandas as pd

from app.app_state import build_filtered_state


class AppStateTests(unittest.TestCase):
    def test_build_filtered_state_filters_frames_and_recomputes_event_metrics(self):
        minute_index = pd.date_range("2026-01-01 00:00", periods=6, freq="h")
        daily_index = pd.to_datetime(["2026-01-01", "2026-01-02"])

        full_master = pd.DataFrame({"value": range(6)}, index=minute_index)
        full_hourly = full_master.copy()
        full_daily = pd.DataFrame({"daily_value": [1, 2]}, index=daily_index)
        full_monthly = pd.DataFrame({"monthly_value": [10]}, index=pd.Index([1]))
        full_weekday = pd.DataFrame({"weekday_value": [20]}, index=pd.Index([2]))
        full_events = pd.DataFrame(
            {
                "timestamp": [minute_index[1], minute_index[4]],
                "chemical": ["Ferric", "HCl"],
                "event_type": ["ON", "OFF"],
            }
        )
        full_all_events = {
            "Ferric_ON": [minute_index[1]],
            "Ferric_OFF": [],
            "HCl_ON": [],
            "HCl_OFF": [minute_index[4]],
        }
        base_event_metrics = pd.DataFrame({"chemical": ["base"]})

        full_state = {
            "master_df": full_master,
            "hourly_df": full_hourly,
            "daily_df": full_daily,
            "monthly_df": full_monthly,
            "weekday_df": full_weekday,
            "events_table": full_events,
            "all_events": full_all_events,
            "event_metrics_df": base_event_metrics,
            "struvite_obs_df": pd.DataFrame({"id": [1]}),
            "chem_labs_df": pd.DataFrame({"id": [2]}),
        }
        window_state = {
            "start_ts": minute_index[1],
            "end_ts": minute_index[3],
        }

        def filter_time_indexed_df(df, start_ts, end_ts):
            return df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()

        def build_period_summaries(daily_df):
            return (
                pd.DataFrame({"summary": [daily_df["daily_value"].sum()]}),
                pd.DataFrame({"summary": [len(daily_df)]}),
            )

        def compute_event_metrics_table(df):
            return pd.DataFrame({"rows_seen": [len(df)]})

        filtered = build_filtered_state(
            full_state,
            window_state,
            filter_time_indexed_df,
            build_period_summaries,
            compute_event_metrics_table,
        )

        self.assertEqual(list(filtered["master_df"].index), list(minute_index[1:4]))
        self.assertEqual(list(filtered["hourly_df"].index), list(minute_index[1:4]))
        self.assertEqual(list(filtered["events_table"]["timestamp"]), [minute_index[1]])
        self.assertEqual(filtered["all_events"]["Ferric_ON"], [minute_index[1]])
        self.assertEqual(filtered["all_events"]["HCl_OFF"], [])
        self.assertEqual(filtered["event_metrics_df"].iloc[0]["rows_seen"], 3)
        self.assertEqual(filtered["monthly_df"].iloc[0]["summary"], 0)
        self.assertEqual(filtered["weekday_df"].iloc[0]["summary"], 0)

    def test_build_filtered_state_preserves_optional_tables(self):
        index = pd.date_range("2026-01-01 00:00", periods=2, freq="h")
        full_state = {
            "master_df": pd.DataFrame({"value": [1, 2]}, index=index),
            "hourly_df": pd.DataFrame({"value": [1, 2]}, index=index),
            "daily_df": pd.DataFrame({"value": [1]}, index=pd.to_datetime(["2026-01-01"])),
            "monthly_df": pd.DataFrame({"value": [1]}),
            "weekday_df": pd.DataFrame({"value": [1]}),
            "events_table": pd.DataFrame(columns=["timestamp", "chemical", "event_type"]),
            "all_events": {},
            "event_metrics_df": pd.DataFrame(),
            "struvite_obs_df": pd.DataFrame({"note": ["scale"]}),
            "chem_labs_df": pd.DataFrame({"note": ["lab"]}),
        }
        window_state = {"start_ts": index[0], "end_ts": index[1]}

        filtered = build_filtered_state(
            full_state,
            window_state,
            lambda df, start_ts, end_ts: df,
            lambda daily_df: (pd.DataFrame(), pd.DataFrame()),
            lambda df: pd.DataFrame(),
        )

        self.assertEqual(filtered["struvite_obs_df"].iloc[0]["note"], "scale")
        self.assertEqual(filtered["chem_labs_df"].iloc[0]["note"], "lab")


if __name__ == "__main__":
    unittest.main()
