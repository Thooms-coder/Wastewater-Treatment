import unittest

import pandas as pd

from scripts.plotting import (
    correlation_heatmap,
    dual_axis_figure,
    event_study_figure,
    multi_panel_figure,
    scatter_with_trend,
)


class PlottingTests(unittest.TestCase):
    def test_dual_axis_figure_builds_two_traces_and_event_marker(self):
        index = pd.date_range("2026-01-01", periods=4, freq="h")
        df = pd.DataFrame(
            {
                "left_signal": [1.0, 2.0, 3.0, 4.0],
                "right_signal": [10.0, 11.0, 12.0, 13.0],
            },
            index=index,
        )
        events = {"Ferric_ON": [index[1]]}

        fig = dual_axis_figure(
            df,
            "left_signal",
            "right_signal",
            "Left",
            "Right",
            "Dual Axis",
            add_events=events,
            rangeslider=False,
        )

        self.assertEqual(len(fig.data), 2)
        self.assertEqual(fig.data[0].name, "Left")
        self.assertEqual(fig.data[1].name, "Right")
        self.assertGreaterEqual(len(fig.layout.shapes), 1)

    def test_dual_axis_figure_focused_scaling_sets_trimmed_ranges(self):
        index = pd.date_range("2026-01-01", periods=6, freq="h")
        df = pd.DataFrame(
            {
                "left_signal": [1.0, 1.1, 1.2, 1.3, 1.4, 50.0],
                "right_signal": [10.0, 10.1, 10.2, 10.3, 10.4, 200.0],
            },
            index=index,
        )

        fig = dual_axis_figure(
            df,
            "left_signal",
            "right_signal",
            "Left",
            "Right",
            "Focused Dual Axis",
            rangeslider=False,
            y1_scale_mode="focused",
            y2_scale_mode="focused",
        )

        self.assertIsNotNone(fig.layout.yaxis.range)
        self.assertIsNotNone(fig.layout.yaxis2.range)
        self.assertLess(fig.layout.yaxis.range[1], 50.0)
        self.assertLess(fig.layout.yaxis2.range[1], 200.0)

    def test_dual_axis_figure_log_scaling_sets_log_axis_for_positive_series(self):
        index = pd.date_range("2026-01-01", periods=4, freq="h")
        df = pd.DataFrame(
            {
                "left_signal": [1.0, 10.0, 100.0, 1000.0],
                "right_signal": [2.0, 20.0, 200.0, 2000.0],
            },
            index=index,
        )

        fig = dual_axis_figure(
            df,
            "left_signal",
            "right_signal",
            "Left",
            "Right",
            "Log Dual Axis",
            rangeslider=False,
            y1_scale_mode="log",
            y2_scale_mode="log",
        )

        self.assertEqual(fig.layout.yaxis.type, "log")
        self.assertEqual(fig.layout.yaxis2.type, "log")

    def test_event_study_figure_contains_iqr_band_and_event_line(self):
        summary = pd.DataFrame(
            {
                "median": [10.0, 8.0, 7.0],
                "q25": [9.0, 7.0, 6.0],
                "q75": [11.0, 9.0, 8.0],
            },
            index=[-60, 0, 60],
        )

        fig = event_study_figure(summary, "Event Study", "NH3")

        self.assertEqual(len(fig.data), 3)
        self.assertEqual(fig.data[1].fill, "tonexty")
        self.assertEqual(fig.data[2].name, "Median")
        self.assertGreaterEqual(len(fig.layout.shapes), 1)

    def test_correlation_heatmap_builds_square_matrix_annotations(self):
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [2.0, 4.0, 6.0],
                "c": [3.0, 2.0, 1.0],
            }
        )

        fig = correlation_heatmap(df, ["a", "b", "c"])

        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].type, "heatmap")
        self.assertEqual(len(fig.layout.annotations), 9)

    def test_scatter_with_trend_adds_regression_line(self):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [2.0, 4.0, 6.0, 8.0],
            }
        )

        fig = scatter_with_trend(df, "x", "y", title="Scatter")

        self.assertEqual(len(fig.data), 2)
        self.assertEqual(fig.data[0].mode, "markers")
        self.assertEqual(fig.data[1].mode, "lines")
        self.assertIn("slope=", fig.data[1].name)

    def test_multi_panel_figure_builds_panel_traces_for_available_events(self):
        window_df = pd.DataFrame(
            {
                "minutes": [-60, 0, 60],
                "y1": [10.0, 9.0, 8.0],
                "y2": [2.0, 3.0, 4.0],
            }
        )
        events = {
            "Ferric OFF": window_df,
            "Ferric ON": window_df,
            "HCl OFF": window_df,
            "HCl ON": window_df,
        }

        fig = multi_panel_figure(None, events, "y1", "y2", "Y1", "Y2", "Panels")

        self.assertEqual(len(fig.data), 8)
        self.assertGreaterEqual(len(fig.layout.shapes), 4)


if __name__ == "__main__":
    unittest.main()
