import unittest
from unittest.mock import patch

import pandas as pd

import app.page_renderers as page_renderers
from scripts.constants import H2S, NH3


class _DummyContext:
    def __init__(self, fake_st=None):
        self._fake_st = fake_st

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getattr__(self, name):
        if self._fake_st is None:
            raise AttributeError(name)
        return getattr(self._fake_st, name)


class _FakeStreamlit:
    def markdown(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def plotly_chart(self, *args, **kwargs):
        return None

    def metric(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def line_chart(self, *args, **kwargs):
        return None

    def bar_chart(self, *args, **kwargs):
        return None

    def download_button(self, *args, **kwargs):
        return None

    def checkbox(self, *args, **kwargs):
        return kwargs.get("value", False)

    def slider(self, *args, **kwargs):
        return kwargs.get("value")

    def selectbox(self, label, options, index=0, **kwargs):
        if options is None:
            return None
        return options[index]

    def radio(self, label, options, index=0, **kwargs):
        return options[index]

    def multiselect(self, label, options, default=None, **kwargs):
        return default if default is not None else options[:2]

    def columns(self, count):
        return [_DummyContext(self) for _ in range(count)]

    def tabs(self, names):
        return [_DummyContext(self) for _ in names]

    def form(self, *args, **kwargs):
        return _DummyContext(self)

    def expander(self, *args, **kwargs):
        return _DummyContext(self)

    def form_submit_button(self, *args, **kwargs):
        return False


class PageRendererSmokeTests(unittest.TestCase):
    def test_all_pages_render_with_fake_streamlit_and_minimal_context(self):
        fake_st = _FakeStreamlit()
        minute_index = pd.date_range("2026-01-01 00:00", periods=220, freq="h")
        daily_index = pd.to_datetime(["2026-01-01", "2026-01-02"])
        base_master = pd.DataFrame(
            {
                NH3: [10.0] * len(minute_index),
                H2S: [2.0] * len(minute_index),
                "total_gpm": [100.0] * len(minute_index),
                "lbs_per_min": [50.0] * len(minute_index),
                "transferred_lbs_vol": [50.0] * len(minute_index),
                "ferric_available": [0] * 72 + [1] * (len(minute_index) - 72),
                "hcl_available": [0] * len(minute_index),
            },
            index=minute_index,
        )
        hourly_df = pd.DataFrame(
            {
                NH3: [10.0, 11.0],
                H2S: [2.0, 2.5],
                "lbs_volatile": [100.0, 110.0],
                "flow_gal_hr": [500.0, 600.0],
                "fecl3_lbs": [4.0, 4.5],
            },
            index=pd.date_range("2026-01-01", periods=2, freq="h"),
        )
        daily_df = pd.DataFrame(
            {
                NH3: [10.0, 11.0],
                H2S: [2.0, 3.0],
                "total_gpm": [100.0, 110.0],
                "transferred_lbs_vol_daily": [1000.0, 1100.0],
                "nh3_coverage": [1.0, 1.0],
                "h2s_coverage": [1.0, 1.0],
                "water_coverage": [1.0, 1.0],
                "n_obs_nh3": [1440, 1440],
                "n_obs_h2s": [1440, 1440],
                "n_obs_water": [1440, 1440],
                "h2s_std": [0.2, 0.3],
                "nh3_std": [0.4, 0.5],
            },
            index=daily_index,
        )
        monthly_df = pd.DataFrame(
            {
                "nh3_monthly_mean": [10.5],
                "h2s_monthly_mean": [2.5],
                "total_gpm_monthly_mean": [105.0],
                "transferred_lbs_vol_monthly_mean": [1050.0],
            },
            index=pd.Index([1]),
        )
        weekday_df = pd.DataFrame(
            {
                "nh3_weekday_mean": [10.5],
                "h2s_weekday_mean": [2.5],
                "total_gpm_weekday_mean": [105.0],
                "transferred_lbs_vol_weekday_mean": [1050.0],
                "weekday_name": ["Monday"],
            },
            index=pd.Index([0]),
        )
        ctx = {
            "master_df": base_master,
            "hourly_df": hourly_df,
            "daily_df": daily_df,
            "monthly_df": monthly_df,
            "weekday_df": weekday_df,
            "event_metrics_df": pd.DataFrame(
                {
                    "chemical": ["Ferric"],
                    "event_type": ["ON"],
                    "signal": ["NH3"],
                    "delta": [-5.0],
                    "percent_change": [-50.0],
                    "time_to_min": [60],
                    "persistence": [120],
                    "post_iqr": [1.0],
                    "n_events": [1],
                }
            ),
            "events_table": pd.DataFrame(
                {
                    "timestamp": [minute_index[72]],
                    "chemical": ["Ferric"],
                    "event_type": ["ON"],
                }
            ),
            "all_events": {"Ferric_ON": [minute_index[72]], "Ferric_OFF": [], "HCl_ON": [], "HCl_OFF": []},
            "start_ts": minute_index[0],
            "end_ts": minute_index[-1],
            "struvite_obs_df": pd.DataFrame({"date": ["2026-01-01"], "location": ["line-a"]}),
            "chem_labs_df": pd.DataFrame({"date": ["2026-01-01"], "pH": [7.0]}),
            "methods_log_df": pd.DataFrame({"date": ["2026-04-19"], "lane": ["Full-scale analytics"]}),
            "thesis_status_text": "# Thesis Outline And Status\n\nCurrent status scaffold is present.",
            "has_data": lambda df, col: df is not None and col in df.columns and df[col].notna().any(),
            "available_columns": lambda df, candidates: [c for c in candidates if c in df.columns],
            "detect_transitions": lambda df, col: ([minute_index[72]], []),
            "compute_event_study_summary": lambda df, chem, event_type, signal: (
                pd.DataFrame({"median": [10.0], "q25": [9.0], "q75": [11.0]}, index=[0]),
                pd.DataFrame({"event_1": [10.0]}, index=[0]),
                True,
            ),
            "add_zscore": lambda df, col, window=1440: pd.Series([0.0] * len(df), index=df.index),
            "detect_anomalies": lambda df, col, threshold=3.0, window=1440: pd.DataFrame(columns=list(df.columns) + ["z_score"]),
            "build_month_labels": lambda index_like: ["Jan" for _ in index_like],
            "metric_value": lambda df, col, fn="mean", fmt="{:.2f}": "1.00",
            "coverage_value": lambda df, col, expected_points: "100.0%",
            "dual_axis_figure": lambda *args, **kwargs: {"kind": "dual"},
            "event_window_figure": lambda *args, **kwargs: {"kind": "window"},
            "event_study_figure": lambda *args, **kwargs: {"kind": "study"},
            "correlation_heatmap": lambda *args, **kwargs: {"kind": "heatmap"},
            "scatter_with_trend": lambda *args, **kwargs: {"kind": "scatter"},
        }

        no_op = lambda *args, **kwargs: None
        with patch.object(page_renderers, "st", fake_st), \
             patch.object(page_renderers, "render_page_header", no_op), \
             patch.object(page_renderers, "render_page_notes", no_op), \
             patch.object(page_renderers, "render_context_band", no_op), \
             patch.object(page_renderers, "render_report_banner", no_op), \
             patch.object(page_renderers, "render_executive_brief", no_op), \
             patch.object(page_renderers, "render_report_highlights", no_op), \
             patch.object(page_renderers, "render_research_alignment", no_op), \
             patch.object(page_renderers, "render_section_intro", no_op), \
             patch.object(page_renderers, "render_help_tip", no_op), \
             patch.object(page_renderers, "render_executive_cards", no_op), \
             patch.object(page_renderers, "render_struvite_placeholder", no_op), \
             patch.object(page_renderers, "multi_panel_figure", lambda *args, **kwargs: {"kind": "panel"}):
            for page in page_renderers.PAGE_OPTIONS:
                with self.subTest(page=page):
                    page_renderers.render_page(page, ctx)


if __name__ == "__main__":
    unittest.main()
