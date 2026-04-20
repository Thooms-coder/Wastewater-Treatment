import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scripts.constants import EVENT_COLUMNS, FLOW, H2S, NH3, RAW_H2S, RAW_NH3, TEMP_H2S, TEMP_NH3, WINDOW_48H
from scripts.plotting import display_label, multi_panel_figure

from app.dashboard_ui import (
    build_chemistry_review_table,
    build_methods_log_template_df,
    build_research_progress_df,
    build_research_alignment_df,
    build_thesis_outline_df,
    metric_value,
    render_summary_cards,
    render_context_band,
    render_executive_brief,
    render_executive_cards,
    render_help_tip,
    render_page_header,
    render_page_notes,
    render_report_banner,
    render_report_highlights,
    render_research_alignment,
    render_section_intro,
    render_struvite_placeholder,
    compute_ferric_mgL_series,
    compute_hcl_mgL_series,
)


PAGE_OPTIONS = [
    "Executive Brief",
    "Operations Review",
    "Chemistry & Dosing",
    "Research Progress",
    "Performance & Coverage",
    "Diagnostics & Data",
]


def render_executive_brief_page(ctx):
    master_df = ctx["master_df"]
    events_table = ctx["events_table"]
    event_metrics_df = ctx["event_metrics_df"]
    all_events = ctx["all_events"]
    start_ts = ctx["start_ts"]
    end_ts = ctx["end_ts"]
    metric_value = ctx["metric_value"]
    coverage_value = ctx["coverage_value"]
    available_columns = ctx["available_columns"]
    dual_axis_figure = ctx["dual_axis_figure"]

    render_page_header(
        "Wastewater Odor Performance Brief",
        "Executive summary of odor conditions, process context, transition activity, and data confidence for the currently selected reporting window.",
    )
    render_page_notes("Executive Brief")
    render_context_band(
        start_ts,
        end_ts,
        len(master_df),
        len(events_table),
        coverage_value(master_df, NH3, len(master_df)),
        coverage_value(master_df, H2S, len(master_df)),
    )
    render_summary_cards(
        [
            {
                "eyebrow": "Window read",
                "title": f"{len(master_df):,} rows with {len(events_table):,} transitions",
                "body": "Start here before looking at individual charts. Event density and window length change what counts as a meaningful pattern.",
                "meta": f"Coverage: NH3 {coverage_value(master_df, NH3, len(master_df))} | H2S {coverage_value(master_df, H2S, len(master_df))}",
            },
            {
                "eyebrow": "Signal level",
                "title": f"NH3 {metric_value(master_df, NH3)} | H2S {metric_value(master_df, H2S)}",
                "body": "These are the top-line odor indicators for the filtered period, using the project’s rolling-window logic for each gas.",
            },
            {
                "eyebrow": "Process context",
                "title": f"{metric_value(master_df, 'total_gpm')} GPM average flow",
                "body": "Use flow and transferred load as the operating context for any odor movement discussed in the reporting narrative.",
                "meta": f"Transferred load: {metric_value(master_df, 'lbs_per_min')} lbs/min",
            },
        ]
    )
    render_report_banner()
    render_executive_brief(master_df, events_table)
    render_report_highlights(master_df, events_table, event_metrics_df)
    render_research_alignment()

    render_section_intro(
        "Key Performance Snapshot",
        "Use these cards as the boardroom summary: current window size, odor levels, operating load, transition activity, and confidence in the underlying sensor coverage.",
    )
    render_help_tip("NH3 is shown as a 15-minute average because ammonia usually behaves more like a sustained background condition. H2S is shown as a 15-minute peak because short sulfur bursts matter operationally and for odor complaints, even when the average stays low.")
    render_executive_cards(
        [
            {
                "label": "Reporting Window",
                "value": f"{master_df.index.min().date()} to {master_df.index.max().date()}",
                "note": f"{len(master_df):,} minute-level rows included in this brief.",
            },
            {
                "label": "Average Odor Load",
                "value": f"NH3 {metric_value(master_df, NH3)} | H2S {metric_value(master_df, H2S)}",
                "note": "Mean window-level odor indicators used for high-level performance review.",
            },
            {
                "label": "Process Throughput",
                "value": f"{metric_value(master_df, 'total_gpm')} GPM",
                "note": f"Average volatile transfer rate: {metric_value(master_df, 'lbs_per_min')} lbs/min.",
            },
            {
                "label": "Detected Chemistry Events",
                "value": f"{len(events_table):,}",
                "note": "Detected chemistry state changes inside the reporting window, such as Ferric or HCl turning ON or OFF.",
            },
        ]
    )
    render_executive_cards(
        [
            {
                "label": "Ferric Activity",
                "value": f"ON {len(all_events.get('Ferric_ON', []))} | OFF {len(all_events.get('Ferric_OFF', []))}",
                "note": "Counts help explain whether odor movement coincides with Ferric changes.",
            },
            {
                "label": "HCl Activity",
                "value": f"ON {len(all_events.get('HCl_ON', []))} | OFF {len(all_events.get('HCl_OFF', []))}",
                "note": "Transition counts summarize HCl operational change frequency.",
            },
            {
                "label": "NH3 Data Confidence",
                "value": coverage_value(master_df, NH3, len(master_df)),
                "note": "Coverage is the share of minute rows with NH3 observations in the window.",
            },
            {
                "label": "H2S Data Confidence",
                "value": coverage_value(master_df, H2S, len(master_df)),
                "note": "Coverage is the share of minute rows with H2S observations in the window.",
            },
        ]
    )

    top_cols = available_columns(master_df, [NH3, H2S, TEMP_NH3, TEMP_H2S, "total_gpm", "lbs_per_min"])
    if top_cols:
        render_section_intro(
            "Headline Trend View",
            "Use one chart to anchor the briefing. This should show the main odor signal against the most relevant operational context for the reporting conversation.",
        )
        st.caption("Select the pairing you want to use as the lead visual in the reporting narrative.")
        render_help_tip("Use `total_gpm` or `transferred_lbs_vol` when you want to compare odor against process load. Use temperature when you want sensor or environmental context. If one signal has extreme spikes, switch the axis mode to `Focused` to trim the most extreme peaks for exploration.")
        with st.form("executive_headline_form"):
            primary_left = st.selectbox("Primary signal", top_cols, index=0)
            primary_right = st.selectbox("Secondary signal", top_cols, index=min(1, len(top_cols) - 1))
            y_scale_mode = st.selectbox("Axis scaling", ["Auto", "Focused", "Log"], index=0)
            st.form_submit_button("Update headline chart")
        fig = dual_axis_figure(
            master_df,
            primary_left,
            primary_right,
            display_label(primary_left),
            display_label(primary_right),
            f"{display_label(primary_left)} vs {display_label(primary_right)}",
            add_events=all_events,
            y1_scale_mode=y_scale_mode.lower(),
            y2_scale_mode=y_scale_mode.lower(),
        )
        st.plotly_chart(fig, use_container_width=True, key="executive_headline_chart")

    render_section_intro(
        "Operational Transition Summary",
        "Use this section to support the narrative with concrete timing and effect estimates for operational changes observed during the reporting window.",
    )
    summary_tabs = st.tabs(["Detected events", "Event response summary"])
    with summary_tabs[0]:
        st.subheader("Detected events")
        st.markdown('<div class="table-caption">Chronological timestamps where Ferric or HCl changed state inside the active filter window.</div>', unsafe_allow_html=True)
        render_help_tip("An event is a detected change in chemistry status at a specific time. `ON` means the signal switched from inactive to active. `OFF` means it switched from active to inactive.")
        st.dataframe(events_table, use_container_width=True, height=260)
    with summary_tabs[1]:
        st.subheader("Event response summary")
        st.markdown('<div class="table-caption">Aggregated pre/post effect metrics computed around matching chemistry change events in the active window.</div>', unsafe_allow_html=True)
        render_help_tip("Each row summarizes what tended to happen before and after a given event type. `delta` is post minus baseline. `percent_change` rescales that difference relative to the baseline. `persistence` is how long the signal stayed below baseline after the event.")
        show_cols = [
            c for c in ["chemical", "event_type", "signal", "delta", "percent_change", "time_to_min", "persistence", "post_iqr", "n_events"]
            if c in event_metrics_df.columns
        ]
        st.dataframe(event_metrics_df[show_cols] if show_cols else event_metrics_df, use_container_width=True, height=260)


def render_operations_review_page(ctx):
    master_df = ctx["master_df"]
    hourly_df = ctx["hourly_df"]
    daily_df = ctx["daily_df"]
    events_table = ctx["events_table"]
    all_events = ctx["all_events"]
    start_ts = ctx["start_ts"]
    end_ts = ctx["end_ts"]
    coverage_value = ctx["coverage_value"]
    has_data = ctx["has_data"]
    detect_transitions = ctx["detect_transitions"]
    compute_event_study_summary = ctx["compute_event_study_summary"]
    dual_axis_figure = ctx["dual_axis_figure"]
    event_window_figure = ctx["event_window_figure"]
    event_study_figure = ctx["event_study_figure"]

    render_page_header(
        "Full Timeline",
        f"Interactive timeline for the filtered window from {start_ts.date()} to {end_ts.date()}, with minute, hourly, and daily views.",
    )
    render_page_notes("Operations Review")
    render_context_band(
        start_ts,
        end_ts,
        len(master_df),
        len(events_table),
        coverage_value(master_df, NH3, len(master_df)),
        coverage_value(master_df, H2S, len(master_df)),
    )
    render_summary_cards(
        [
            {
                "eyebrow": "Best use",
                "title": "Read the page in three passes",
                "body": "Start with a timeline, then inspect a specific transition, then check whether the same response repeats across events.",
            },
            {
                "eyebrow": "Current density",
                "title": f"{len(events_table):,} transitions inside the selected window",
                "body": "Higher event density supports repeated-event reading. Sparse windows usually need more careful single-event inspection.",
            },
            {
                "eyebrow": "Interaction cue",
                "title": "Change the time scale before changing the story",
                "body": "If a pattern disappears when you move from minute to hourly or daily views, it may be a spike rather than an operating regime change.",
            },
        ]
    )

    timeline_tab, events_tab, comparison_tab = st.tabs(["Timeline", "Event windows", "Comparisons"])

    with timeline_tab:
        render_section_intro(
            "Interactive Timeline Builder",
            "Adjust resolution and signal pairing to match the scale of the question. Short spikes, hourly context, and longer trends should not be read from the same view.",
        )
        render_help_tip("Choose `1-minute` for short spikes, `1-hour` for operating context, and `Daily` for management-level trend summaries. Use `Focused` axis scaling when rare peaks flatten the rest of the chart.")
        with st.form("ops_timeline_form"):
            resolution = st.radio("Resolution", ["1-minute", "1-hour", "Daily"], horizontal=True)
            if resolution == "1-minute":
                active_df = master_df
            elif resolution == "1-hour":
                active_df = hourly_df if hourly_df is not None else master_df
            else:
                active_df = daily_df if daily_df is not None else master_df

            cols = [c for c in active_df.columns if pd.api.types.is_numeric_dtype(active_df[c])]
            default_left = NH3 if NH3 in cols else cols[0]
            default_right = H2S if H2S in cols else cols[min(1, len(cols) - 1)]

            left_col = st.selectbox("Primary y-axis", cols, index=cols.index(default_left) if default_left in cols else 0)
            right_col = st.selectbox("Secondary y-axis", cols, index=cols.index(default_right) if default_right in cols else 0)
            y_scale_mode = st.selectbox("Axis scaling", ["Auto", "Focused", "Log"], index=0)
            show_events = st.checkbox("Overlay transition markers", value=True)
            st.form_submit_button("Update timeline")
        st.caption(
            "Use the overlay when you are testing whether odor changes line up with Ferric or HCl state changes. "
            "Turn it off when the chart becomes too visually dense."
        )

        fig = dual_axis_figure(
            active_df,
            left_col,
            right_col,
            display_label(left_col),
            display_label(right_col),
            f"{display_label(left_col)} vs {display_label(right_col)} ({resolution})",
            add_events=all_events if show_events else None,
            bar_second=(resolution == "1-hour" and right_col in ["flow_gal_hr", "lbs_volatile", "fecl3_lbs"]),
            y1_scale_mode=y_scale_mode.lower(),
            y2_scale_mode=y_scale_mode.lower(),
        )
        st.plotly_chart(fig, use_container_width=True, key="ops_timeline_chart")

        render_section_intro(
            "Script-Aligned Shortcuts",
            "These views mirror common analysis pairings used elsewhere in the project so you can quickly reproduce familiar reads without rebuilding the chart setup each time.",
        )
        shortcut_tabs = st.tabs(["NH3 and H2S", "H2S and temperature", "Load context"])
        with shortcut_tabs[0]:
            st.plotly_chart(dual_axis_figure(master_df, NH3, H2S, "NH₃ (ppm)", "H₂S (ppm)", "NH₃ and H₂S – Full Timeline", add_events=all_events), use_container_width=True, key="ops_shortcut_nh3_h2s_full")
            if hourly_df is not None and has_data(hourly_df, NH3) and has_data(hourly_df, H2S):
                st.plotly_chart(dual_axis_figure(hourly_df, NH3, H2S, "NH₃ (ppm) — hourly avg", "H₂S (ppm) — hourly max", "NH₃ and H₂S – Hourly", add_events=all_events), use_container_width=True, key="ops_shortcut_nh3_h2s_hourly")
        with shortcut_tabs[1]:
            if has_data(master_df, H2S) and has_data(master_df, TEMP_H2S):
                st.plotly_chart(dual_axis_figure(master_df, H2S, TEMP_H2S, "H₂S (ppm)", "Temperature (°F)", "H₂S and Temperature – Full Timeline", add_events=all_events), use_container_width=True, key="ops_shortcut_h2s_temp")
            else:
                st.info("H2S and temperature view is not available in the current window.")
        with shortcut_tabs[2]:
            if hourly_df is not None and "lbs_volatile" in hourly_df.columns:
                hourly_plot = hourly_df[["lbs_volatile"]].rename(columns={"lbs_volatile": "Transferred Lbs Vol"})
                merged = master_df[[H2S]].join(hourly_plot, how="outer")
                st.plotly_chart(dual_axis_figure(merged, H2S, "Transferred Lbs Vol", "H₂S (ppm)", "Transferred volatile solids (lbs/hr)", "H₂S and hourly transferred volatile solids", add_events=all_events, bar_second=True), use_container_width=True, key="ops_shortcut_load_context")
            else:
                st.info("Load-context shortcut is not available in the current window.")

    with events_tab:
        render_section_intro(
            "Single Transition Inspection",
            "Use this section when leadership needs a concrete before-and-after view around one specific chemistry event rather than a full-window trend.",
        )
        render_help_tip("The vertical marker at minute 0 is the event time: the moment Ferric or HCl changed state. Negative values are pre-event; positive values are post-event.")
        with st.form("ops_single_event_form"):
            event_family = st.selectbox("Event family", list(EVENT_COLUMNS.keys()), key="ops_event_family")
            event_direction = st.radio("Transition", ["ON", "OFF"], horizontal=True, key="ops_event_direction")
            signal_mode = st.radio(
                "Window view",
                ["NH3 vs H2S", "NH3 vs Temperature", "H2S vs Temperature", "NH3 vs Load", "H2S vs Load"],
                horizontal=True,
                key="ops_signal_mode",
            )
            st.form_submit_button("Inspect event window")
        on_events, off_events = detect_transitions(master_df, EVENT_COLUMNS[event_family])
        event_times = on_events if event_direction == "ON" else off_events
        if len(event_times) == 0:
            st.warning("No events found for this selection.")
        else:
            st.metric("Available events", len(event_times))
            with st.form("ops_single_event_timestamp_form"):
                event_time = st.selectbox("Event timestamp", list(event_times), format_func=lambda x: x.strftime("%Y-%m-%d %H:%M"), key="ops_event_time")
                st.form_submit_button("Load selected event")
            window_df = master_df.loc[event_time - WINDOW_48H : event_time + WINDOW_48H].copy()
            window_df["minutes_from_event"] = (window_df.index - event_time).total_seconds() / 60
            pairs = {
                "NH3 vs H2S": (NH3, H2S, "NH₃ (ppm)", "H₂S (ppm)", False),
                "NH3 vs Temperature": (NH3, TEMP_NH3, "NH₃ (ppm)", "Temperature (°F)", False),
                "H2S vs Temperature": (H2S, TEMP_H2S, "H₂S (ppm)", "Temperature (°F)", False),
                "NH3 vs Load": (NH3, "transferred_lbs_vol", "NH₃ (ppm)", "Transferred Vol (lbs/min equiv)", True),
                "H2S vs Load": (H2S, "transferred_lbs_vol", "H₂S (ppm)", "Transferred Vol (lbs/min equiv)", True),
            }
            y1, y2, l1, l2, bar = pairs[signal_mode]
            st.plotly_chart(
                event_window_figure(window_df, y1, y2, l1, l2, f"{signal_mode} around {event_family} {event_direction} event", bar=bar),
                use_container_width=True,
                key="ops_single_event_window",
            )

        render_section_intro(
            "Repeated Event Response",
            "This view summarizes whether similar chemistry events tend to produce a repeatable odor response across the reporting window.",
        )
        render_help_tip("The median line shows the typical aligned response across events of the same type, such as Ferric ON or HCl OFF. The shaded band shows how much those responses varied from event to event.")
        with st.form("ops_repeated_event_form"):
            s1, s2, s3 = st.columns(3)
            chem = s1.selectbox("Chemical", list(EVENT_COLUMNS.keys()), key="ops_study_chem")
            event_type = s2.selectbox("Event type", ["ON", "OFF"], key="ops_study_type")
            signal_label = s3.selectbox("Signal", ["NH3", "H2S"], key="ops_study_signal")
            st.form_submit_button("Update event study")
        signal_col = NH3 if signal_label == "NH3" else H2S
        summary, aligned_df, pretrend_ok = compute_event_study_summary(master_df, chem, event_type, signal_col)
        if summary is None or summary.empty:
            st.warning("No aligned event windows were available for this selection.")
        else:
            st.plotly_chart(
                event_study_figure(summary, f"{signal_label} Response Around {chem} {event_type}", f"{signal_label} (ppm)"),
                use_container_width=True,
                key="ops_repeated_event_study",
            )
            render_summary_cards(
                [
                    {
                        "eyebrow": "Events aligned",
                        "title": f"{aligned_df.shape[1]}",
                        "body": "Number of matched events used to build the aligned response summary.",
                    },
                    {
                        "eyebrow": "Median at event",
                        "title": f"{summary.loc[0, 'median']:.2f}" if 0 in summary.index else "NA",
                        "body": "Typical signal level at the event time after aligning all selected events.",
                    },
                    {
                        "eyebrow": "Pretrend stable",
                        "title": "Yes" if pretrend_ok else "No",
                        "body": "Whether the pre-event trend looked sufficiently flat before alignment.",
                    },
                    {
                        "eyebrow": "Window",
                        "title": "±72h",
                        "body": "Event-study window used to summarize the aligned response.",
                    },
                ]
            )

    with comparison_tab:
        render_section_intro(
            "Operational Transition Summary",
            "Use this section to support the narrative with concrete timing and effect estimates for operational changes observed during the reporting window.",
        )
        comparison_summary_tabs = st.tabs(["Detected events", "Event response summary"])
        with comparison_summary_tabs[0]:
            st.subheader("Detected events")
            st.markdown('<div class="table-caption">Chronological timestamps where Ferric or HCl changed state inside the active filter window.</div>', unsafe_allow_html=True)
            render_help_tip("An event is a detected change in chemistry status at a specific time. `ON` means the signal switched from inactive to active. `OFF` means it switched from active to inactive.")
            st.dataframe(events_table, use_container_width=True, height=260)
        with comparison_summary_tabs[1]:
            st.subheader("Event response summary")
            st.markdown('<div class="table-caption">Aggregated pre/post effect metrics computed around matching chemistry change events in the active window.</div>', unsafe_allow_html=True)
            render_help_tip("Each row summarizes what tended to happen before and after a given event type. `delta` is post minus baseline. `percent_change` rescales that difference relative to the baseline. `persistence` is how long the signal stayed below baseline after the event.")
            show_cols = [
                c for c in ["chemical", "event_type", "signal", "delta", "percent_change", "time_to_min", "persistence", "post_iqr", "n_events"]
                if c in ctx["event_metrics_df"].columns
            ]
            event_metrics_df = ctx["event_metrics_df"]
            st.dataframe(event_metrics_df[show_cols] if show_cols else event_metrics_df, use_container_width=True, height=260)

        render_section_intro(
            "Cross-Transition Comparison",
            "Use side-by-side windows to compare whether Ferric and HCl changes produce similar or distinct operational signatures.",
        )
        compare_options = {
        "NH3 vs H2S": (NH3, H2S, "NH₃ (ppm)", "H₂S (ppm)"),
        "NH3 vs Temperature": (NH3, TEMP_NH3, "NH₃ (ppm)", "Temperature (°F)"),
        "H2S vs Temperature": (H2S, TEMP_H2S, "H₂S (ppm)", "Temperature (°F)"),
        "NH3 vs Sludge Flow": (NH3, FLOW, "NH₃ (ppm)", "Sludge Flow (GPM)"),
        "H2S vs Sludge Flow": (H2S, FLOW, "H₂S (ppm)", "Sludge Flow (GPM)"),
        }
        with st.form("ops_compare_form"):
            choice = st.selectbox("Comparison view", list(compare_options.keys()), key="ops_compare_choice")
            st.form_submit_button("Update comparison")
        y1, y2, y1_label, y2_label = compare_options[choice]
        compare_events = {}
        for chem_name, col in EVENT_COLUMNS.items():
            on_events, off_events = detect_transitions(master_df, col)
            if len(off_events) > 0:
                compare_events[f"{chem_name} OFF"] = off_events[0]
            if len(on_events) > 0:
                compare_events[f"{chem_name} ON"] = on_events[0]
        ordered = {k: compare_events[k] for k in ["Ferric OFF", "Ferric ON", "HCl OFF", "HCl ON"] if k in compare_events}
        if len(ordered) == 0:
            st.warning("No transition windows available.")
        else:
            event_windows = {}
            for event_name, center in ordered.items():
                w = master_df.loc[center - WINDOW_48H : center + WINDOW_48H].copy()
                if not w.empty:
                    w["minutes"] = (w.index - center).total_seconds() / 60
                event_windows[event_name] = w
            st.plotly_chart(
                multi_panel_figure(master_df, event_windows, y1, y2, y1_label, y2_label, f"{choice} Across Operational Transitions"),
                use_container_width=True,
                key="ops_transition_comparison_panels",
            )


def render_chemistry_dosing_page(ctx):
    master_df = ctx["master_df"]
    events_table = ctx["events_table"]
    all_events = ctx["all_events"]
    start_ts = ctx["start_ts"]
    end_ts = ctx["end_ts"]
    struvite_obs_df = ctx["struvite_obs_df"]
    chem_labs_df = ctx["chem_labs_df"]
    coverage_value = ctx["coverage_value"]
    available_columns = ctx["available_columns"]
    dual_axis_figure = ctx["dual_axis_figure"]

    render_page_header(
        "Chemistry & Dosing",
        "Review chemistry dose intensity in mg/L, keep feed context in lb/day, and track what is still missing for HCl optimization and struvite analysis.",
    )
    render_page_notes("Chemistry & Dosing")
    render_context_band(
        start_ts,
        end_ts,
        len(master_df),
        len(events_table),
        coverage_value(master_df, NH3, len(master_df)),
        coverage_value(master_df, H2S, len(master_df)),
    )
    render_summary_cards(
        [
            {
                "eyebrow": "Dose intensity",
                "title": f"Ferric {metric_value(pd.DataFrame({'ferric_active_mg_per_L': compute_ferric_mgL_series(master_df)}), 'ferric_active_mg_per_L')} mg/L median",
                "body": "Read ferric dose intensity in mg/L first so chemistry stays comparable as plant flow changes; use lb/day only as supporting feed context.",
            },
            {
                "eyebrow": "Acid context",
                "title": f"HCl {metric_value(pd.DataFrame({'hcl_active_mg_per_L': compute_hcl_mgL_series(master_df)}), 'hcl_active_mg_per_L')} mg/L median",
                "body": "HCl is shown as dose intensity in the same mg/L frame so dose-response patterns can be read directly without mentally normalizing for flow.",
            },
            {
                "eyebrow": "Maturity",
                "title": "Useful for interpretation, not yet the source of truth",
                "body": "The chemistry layer is still assumption-backed in places. Measured feed logs and lab tables should supersede representative dose logic.",
            },
        ]
    )

    ferric_mgl = compute_ferric_mgL_series(master_df)
    hcl_mgl = compute_hcl_mgL_series(master_df)
    chemistry_cols = available_columns(
        master_df,
        [
            NH3, H2S, "total_gpm", "lbs_per_min",
            "ferric_solution_lbs_per_day", "ferric_active_lbs_per_day",
            "hcl_solution_lbs_per_day", "hcl_active_lbs_per_day",
        ],
    )
    chemistry_df = master_df[chemistry_cols].copy() if chemistry_cols else pd.DataFrame(index=master_df.index)
    if not ferric_mgl.empty:
        chemistry_df = chemistry_df.join(ferric_mgl, how="left")
    if not hcl_mgl.empty:
        chemistry_df = chemistry_df.join(hcl_mgl, how="left")

    overview_tab, coverage_tab, inputs_tab = st.tabs(["Dose overview", "Research coverage", "Optional inputs"])

    with overview_tab:
        render_section_intro(
            "Chemistry Workflow Status",
            "This section surfaces the chemistry-oriented fields already present in the repo so the app can be evaluated against the research notes rather than only against the odor UI.",
        )
        render_help_tip("Ferric and HCl dose-intensity features are contextual right now. They are useful for interpretation, but measured feed logs should replace representative assumptions.")
        with st.expander("View chemistry workflow table", expanded=True):
            st.dataframe(build_chemistry_review_table(master_df), use_container_width=True, height=220)

        render_section_intro(
            "Dose And Odor Timeline",
            "This chart is the first step toward the optimization workflow described in the notes: compare chemistry dose intensity, odor response, and operating load inside the same window.",
        )
        render_help_tip("Dose intensity is presented in mg/L first because that is the clearest engineering concentration view. `lb/day` remains available as secondary feed-rate context. `ferric_active_mg_per_L` and `hcl_active_mg_per_L` are computed from active lb/day and flow-derived MGD using the standard 8.34 conversion. If a few dose spikes flatten the chart, switch axis scaling to `Focused`.")
        chemistry_plot_options = [
            c for c in [
                NH3, H2S,
                "ferric_active_mg_per_L", "hcl_active_mg_per_L",
                "ferric_active_lbs_per_day", "hcl_active_lbs_per_day",
                "ferric_solution_lbs_per_day", "hcl_solution_lbs_per_day",
                "total_gpm", "lbs_per_min",
            ] if c in chemistry_df.columns
        ]
        if len(chemistry_plot_options) < 2:
            st.info("Chemistry plotting requires at least two available chemistry or operating columns in the current window.")
        else:
            default_left_col = "ferric_active_mg_per_L" if ("ferric_active_mg_per_L" in chemistry_plot_options and chemistry_df.get("ferric_active_mg_per_L", pd.Series(dtype=float)).notna().any()) else None
            if default_left_col is None and "ferric_active_lbs_per_day" in chemistry_plot_options:
                default_left_col = "ferric_active_lbs_per_day"
            if default_left_col is None:
                default_left_col = chemistry_plot_options[0]

            default_right_col = None
            if "hcl_active_mg_per_L" in chemistry_plot_options and chemistry_df.get("hcl_active_mg_per_L", pd.Series(dtype=float)).notna().any():
                default_right_col = "hcl_active_mg_per_L"
            elif "hcl_active_lbs_per_day" in chemistry_plot_options and chemistry_df.get("hcl_active_lbs_per_day", pd.Series(dtype=float)).notna().any():
                default_right_col = "hcl_active_lbs_per_day"
            elif len(chemistry_plot_options) > 1:
                default_right_col = chemistry_plot_options[1]
            else:
                default_right_col = chemistry_plot_options[0]

            default_left = chemistry_plot_options.index(default_left_col)
            default_right = chemistry_plot_options.index(default_right_col)
            if default_right == default_left and len(chemistry_plot_options) > 1:
                default_right = 1 if default_left == 0 else 0
            with st.form("chemistry_plot_form"):
                left_col = st.selectbox("Primary chemistry signal", chemistry_plot_options, index=default_left, key="chem_left")
                right_col = st.selectbox("Secondary chemistry signal / context", chemistry_plot_options, index=default_right, key="chem_right")
                y_scale_mode = st.selectbox("Axis scaling", ["Auto", "Focused", "Log"], index=1, key="chem_scale")
                st.form_submit_button("Update chemistry view")
            missing_in_window = [display_label(col) for col in [left_col, right_col] if col in chemistry_df.columns and not chemistry_df[col].notna().any()]
            if missing_in_window:
                st.warning(f"No plottable values in the current window for: {', '.join(missing_in_window)}.")
            elif "hcl_active_mg_per_L" in {left_col, right_col} and not chemistry_df["hcl_active_mg_per_L"].notna().any():
                st.info("HCl dose intensity (mg/L) is empty in this window because flow-normalized conversion requires non-zero `total_gpm`. Switch to `HCl active feed context (lb/day)` to view the assumed HCl series directly.")
            st.plotly_chart(
                dual_axis_figure(
                    chemistry_df,
                    left_col,
                    right_col,
                    display_label(left_col),
                    display_label(right_col),
                    f"{display_label(left_col)} vs {display_label(right_col)}",
                    add_events=all_events,
                    bar_second=right_col in {
                        "ferric_solution_lbs_per_day", "ferric_active_lbs_per_day",
                        "hcl_solution_lbs_per_day", "hcl_active_lbs_per_day",
                    },
                    y1_scale_mode=y_scale_mode.lower(),
                    y2_scale_mode=y_scale_mode.lower(),
                ),
                use_container_width=True,
                key="chemistry_timeline_chart",
            )

    with coverage_tab:
        render_section_intro(
            "Research Objective Coverage",
            "This keeps the chemistry page tied to the dissertation plan and shows which chemistry questions are already supported versus still pending.",
        )
        with st.expander("View research alignment table", expanded=True):
            st.dataframe(build_research_alignment_df(), use_container_width=True, height=240)

    with inputs_tab:
        render_struvite_placeholder(struvite_obs_df, chem_labs_df)


def render_research_progress_page(ctx):
    master_df = ctx["master_df"]
    events_table = ctx["events_table"]
    start_ts = ctx["start_ts"]
    end_ts = ctx["end_ts"]
    methods_log_df = ctx.get("methods_log_df")
    thesis_status_text = ctx.get("thesis_status_text", "")
    coverage_value = ctx["coverage_value"]

    render_page_header(
        "Research Progress",
        "Track the thesis program across bench-scale method development, full-scale analytics, scaling observations, and writing progress.",
    )
    render_page_notes("Research Progress")
    render_context_band(
        start_ts,
        end_ts,
        len(master_df),
        len(events_table),
        coverage_value(master_df, NH3, len(master_df)),
        coverage_value(master_df, H2S, len(master_df)),
    )
    render_summary_cards(
        [
            {
                "eyebrow": "Program view",
                "title": "Full-scale analysis is ahead of the other lanes",
                "body": "Use this page to make the imbalance explicit instead of letting the polished analytics UI imply the whole thesis is equally mature.",
            },
            {
                "eyebrow": "Bench scale",
                "title": "Method development still needs continuous logging",
                "body": "The committee asked for failed methods, pivots, and blocked paths to be captured as part of progress, not edited out later.",
            },
            {
                "eyebrow": "Writing",
                "title": "Thesis drafting should move in parallel",
                "body": "Introduction, methods, and status notes should accumulate while experiments and plant analysis continue.",
            },
        ]
    )

    milestones_tab, methods_tab, writing_tab = st.tabs(["Milestones", "Methods log", "Writing status"])

    with milestones_tab:
        render_section_intro(
            "Milestone Tracker",
            "This table turns the committee guidance into a working project board so active full-scale work, blocked bench-scale work, and writing obligations can be tracked together.",
        )
        render_help_tip("Use this as the thesis-program checklist. A technically polished dashboard does not mean every research lane is equally mature.")
        with st.expander("View milestone tracker", expanded=True):
            st.dataframe(build_research_progress_df(), use_container_width=True, height=260)

    with methods_tab:
        render_section_intro(
            "Methods Log Workflow",
            "The committee explicitly asked for failed methods and experimental pivots to be documented. This section is the working structure for that log.",
        )
        render_help_tip("Create `notes/methods_log.csv` from the template and update it continuously. Include failed meter setups, chemistry attempts, and why a method changed.")
        if methods_log_df is not None and not methods_log_df.empty:
            st.caption("Using active methods log from `notes/methods_log.csv`.")
            with st.expander("View methods log", expanded=True):
                st.dataframe(methods_log_df, use_container_width=True, height=240)
        else:
            st.caption("No live methods log found yet. Showing starter template from `notes/methods_log_template.csv`.")
            with st.expander("View methods log template", expanded=True):
                st.dataframe(build_methods_log_template_df(), use_container_width=True, height=220)

    with writing_tab:
        render_section_intro(
            "Thesis Outline And Status",
            "This scaffold keeps the writing work moving in parallel with the experiments and plant analysis instead of waiting for perfect final results.",
        )
        render_help_tip("Introduction, literature review, methods, and failed paths should accumulate as the project progresses.")
        with st.expander("View thesis outline tracker", expanded=True):
            st.dataframe(build_thesis_outline_df(), use_container_width=True, height=250)
        if thesis_status_text.strip():
            st.markdown(thesis_status_text)
        else:
            st.info("No thesis status markdown found. Add `notes/thesis_outline_status.md` to track the writing plan in the repo.")


def render_performance_coverage_page(ctx):
    master_df = ctx["master_df"]
    hourly_df = ctx["hourly_df"]
    daily_df = ctx["daily_df"]
    monthly_df = ctx["monthly_df"]
    weekday_df = ctx["weekday_df"]
    events_table = ctx["events_table"]
    start_ts = ctx["start_ts"]
    end_ts = ctx["end_ts"]
    coverage_value = ctx["coverage_value"]
    available_columns = ctx["available_columns"]
    build_month_labels = ctx["build_month_labels"]
    dual_axis_figure = ctx["dual_axis_figure"]
    correlation_heatmap = ctx["correlation_heatmap"]
    scatter_with_trend = ctx["scatter_with_trend"]

    render_page_header(
        "Aggregates & Coverage",
        "Daily, monthly, weekday, and coverage views recalculated from the current time filter so you can understand both signal level and data completeness.",
    )
    st.caption("Aggregate views are recalculated from the currently filtered daily window.")
    render_page_notes("Performance & Coverage")
    render_context_band(
        start_ts,
        end_ts,
        len(master_df),
        len(events_table),
        coverage_value(master_df, NH3, len(master_df)),
        coverage_value(master_df, H2S, len(master_df)),
    )
    render_summary_cards(
        [
            {
                "eyebrow": "Use case",
                "title": "Switch here when minute-level plots feel too noisy",
                "body": "Daily, monthly, and weekday compressions help separate recurring structure from one-off excursions.",
            },
            {
                "eyebrow": "Data confidence",
                "title": f"NH3 {coverage_value(master_df, NH3, len(master_df))} | H2S {coverage_value(master_df, H2S, len(master_df))}",
                "body": "Coverage remains part of the story even when the plots are aggregated.",
            },
            {
                "eyebrow": "Interaction cue",
                "title": "Read aggregates before correlations",
                "body": "Make sure the daily and coverage views are stable before trusting any heatmap or scatter relationship.",
            },
        ]
    )
    render_section_intro(
        "Compressed Views",
        "Switch here when the minute-level timeline is too granular and you need daily structure, seasonal shape, weekday patterns, or a direct read on missingness.",
    )
    render_help_tip("Coverage charts tell you how much data was actually present. Low coverage can make averages and event summaries look more certain than they are.")

    tabs = st.tabs(["Daily", "Monthly", "Weekday", "Coverage"])

    with tabs[0]:
        if daily_df is None:
            st.info("Daily dataset not found.")
        else:
            cols = available_columns(daily_df, [NH3, H2S, "total_gpm", "transferred_lbs_vol_daily", "h2s_std", "nh3_std"])
            with st.form("perf_daily_form"):
                left = st.selectbox("Daily primary", cols, key="daily_left")
                right = st.selectbox("Daily secondary", cols, index=min(1, len(cols) - 1), key="daily_right")
                y_scale_mode = st.selectbox("Axis scaling", ["Auto", "Focused", "Log"], index=0, key="daily_scale")
                st.form_submit_button("Update daily view")
            st.plotly_chart(
                dual_axis_figure(
                    daily_df,
                    left,
                    right,
                    display_label(left),
                    display_label(right),
                    f"Daily {display_label(left)} vs {display_label(right)}",
                    y1_scale_mode=y_scale_mode.lower(),
                    y2_scale_mode=y_scale_mode.lower(),
                ),
                use_container_width=True,
                key="perf_daily_chart",
            )
            with st.expander("View daily aggregate table", expanded=False):
                st.dataframe(daily_df[cols + available_columns(daily_df, ["n_obs_nh3", "n_obs_h2s", "n_obs_water", "nh3_coverage", "h2s_coverage", "water_coverage"])], use_container_width=True, height=280)

    with tabs[1]:
        if monthly_df is None:
            st.info("Monthly summary not found.")
        else:
            display_monthly = monthly_df.copy()
            display_monthly.index = build_month_labels(display_monthly.index)
            monthly_cols = available_columns(display_monthly, ["nh3_monthly_mean", "h2s_monthly_mean", "total_gpm_monthly_mean", "transferred_lbs_vol_monthly_mean"])
            st.caption("Monthly averages by selected reporting window month.")
            monthly_fig = go.Figure()
            for col in monthly_cols:
                monthly_fig.add_trace(
                    go.Bar(
                        x=list(display_monthly.index),
                        y=display_monthly[col],
                        name=display_label(col),
                    )
                )
            monthly_fig.update_layout(
                title="Monthly summary",
                barmode="group",
                xaxis_title="Month",
                yaxis_title="Value",
                hovermode="x unified",
                legend=dict(orientation="h"),
                template="plotly_white",
            )
            st.plotly_chart(monthly_fig, use_container_width=True, key="perf_monthly_chart")
            with st.expander("View monthly summary table", expanded=False):
                st.dataframe(display_monthly, use_container_width=True, height=280)

    with tabs[2]:
        if weekday_df is None:
            st.info("Weekday summary not found.")
        else:
            display_weekday = weekday_df.copy()
            if "weekday_name" in display_weekday.columns:
                display_weekday = display_weekday.set_index("weekday_name")
            weekday_cols = available_columns(display_weekday, ["nh3_weekday_mean", "h2s_weekday_mean", "total_gpm_weekday_mean", "transferred_lbs_vol_weekday_mean"])
            st.caption("Weekday averages across the selected reporting window.")
            weekday_fig = go.Figure()
            for col in weekday_cols:
                weekday_fig.add_trace(
                    go.Bar(
                        x=list(display_weekday.index),
                        y=display_weekday[col],
                        name=display_label(col),
                    )
                )
            weekday_fig.update_layout(
                title="Weekday summary",
                barmode="group",
                xaxis_title="Weekday",
                yaxis_title="Value",
                hovermode="x unified",
                legend=dict(orientation="h"),
                template="plotly_white",
            )
            st.plotly_chart(weekday_fig, use_container_width=True, key="perf_weekday_chart")
            with st.expander("View weekday summary table", expanded=False):
                st.dataframe(display_weekday, use_container_width=True, height=280)

    with tabs[3]:
        if daily_df is None:
            st.info("Coverage metrics require master_daily.parquet.")
        else:
            coverage_cols = available_columns(daily_df, ["nh3_coverage", "h2s_coverage", "water_coverage"])
            if coverage_cols:
                st.caption("Daily coverage share by signal. Higher values mean more complete daily records.")
                coverage_fig = go.Figure()
                for col in coverage_cols:
                    coverage_fig.add_trace(
                        go.Scatter(
                            x=daily_df.index,
                            y=daily_df[col],
                            mode="lines",
                            name=display_label(col),
                        )
                    )
                coverage_fig.update_layout(
                    title="Daily data coverage",
                    xaxis_title="Date",
                    yaxis_title="Coverage share",
                    hovermode="x unified",
                    legend=dict(orientation="h"),
                    template="plotly_white",
                )
                st.plotly_chart(coverage_fig, use_container_width=True, key="perf_coverage_chart")
                with st.expander("View coverage detail table", expanded=False):
                    st.dataframe(daily_df[coverage_cols + available_columns(daily_df, ["n_obs_nh3", "n_obs_h2s", "n_obs_water"])], use_container_width=True, height=260)

    render_section_intro(
        "Relationship Screening",
        "Use the correlation matrix and scatter view to support a performance narrative with simple relationship checks between odor, throughput, and operating context.",
    )
    render_help_tip("Correlation shows linear association, not causation. Use the scatter plot to check whether the relationship is real, clustered, or driven by a few extreme points.")
    analysis_df = daily_df.copy() if daily_df is not None else master_df.resample("1D").mean(numeric_only=True)
    analysis_df = analysis_df.copy()
    numeric_cols = [c for c in analysis_df.columns if pd.api.types.is_numeric_dtype(analysis_df[c])]
    default_corr = [c for c in [NH3, H2S, "total_gpm", "transferred_lbs_vol_daily", "transferred_lbs_vol", "nh3_std", "h2s_std", "ferric_active_lbs_per_day"] if c in numeric_cols]
    with st.form("perf_relationships_form"):
        selected = st.multiselect("Columns for heatmap", numeric_cols, default=default_corr[: min(len(default_corr), 8)], key="perf_heatmap_cols")
        scatter_source = hourly_df if hourly_df is not None else analysis_df
        scatter_cols = [c for c in scatter_source.columns if pd.api.types.is_numeric_dtype(scatter_source[c])]
        x_default = "lbs_volatile" if "lbs_volatile" in scatter_cols else scatter_cols[0]
        y_default = H2S if H2S in scatter_cols else scatter_cols[min(1, len(scatter_cols) - 1)]
        x_col = st.selectbox("Scatter x", scatter_cols, index=scatter_cols.index(x_default) if x_default in scatter_cols else 0, key="perf_scatter_x")
        y_col = st.selectbox("Scatter y", scatter_cols, index=scatter_cols.index(y_default) if y_default in scatter_cols else 0, key="perf_scatter_y")
        color_col = st.selectbox("Color by (optional)", [None] + scatter_cols, index=0, key="perf_scatter_color")
        st.form_submit_button("Update relationship screening")
    if len(selected) >= 2:
        st.plotly_chart(correlation_heatmap(analysis_df, selected, title="Correlation heatmap of selected metrics"), use_container_width=True, key="perf_correlation_heatmap")
    else:
        st.info("Select at least two columns for the correlation heatmap.")
    scatter_title = f"{display_label(y_col)} vs {display_label(x_col)}"
    st.plotly_chart(scatter_with_trend(scatter_source, x_col, y_col, color_col=color_col, title=scatter_title), use_container_width=True, key="perf_scatter_trend")


def render_diagnostics_data_page(ctx):
    master_df = ctx["master_df"]
    hourly_df = ctx["hourly_df"]
    daily_df = ctx["daily_df"]
    monthly_df = ctx["monthly_df"]
    weekday_df = ctx["weekday_df"]
    events_table = ctx["events_table"]
    event_metrics_df = ctx["event_metrics_df"]
    start_ts = ctx["start_ts"]
    end_ts = ctx["end_ts"]
    coverage_value = ctx["coverage_value"]
    available_columns = ctx["available_columns"]
    add_zscore = ctx["add_zscore"]
    detect_anomalies = ctx["detect_anomalies"]

    render_page_header(
        "Anomalies",
        "Use rolling z-scores to identify unusual observations in odor, temperature, flow, or load-normalized signals within the filtered study window.",
    )
    render_page_notes("Diagnostics & Data")
    render_context_band(
        start_ts,
        end_ts,
        len(master_df),
        len(events_table),
        coverage_value(master_df, NH3, len(master_df)),
        coverage_value(master_df, H2S, len(master_df)),
    )
    render_summary_cards(
        [
            {
                "eyebrow": "Anomaly logic",
                "title": "Flags are relative to recent history, not absolute limits",
                "body": "A spike only counts as unusual if it departs enough from the rolling baseline selected in the controls.",
            },
            {
                "eyebrow": "Best use",
                "title": "Use this page for triage and raw verification",
                "body": "Start with anomaly detection when you need to find candidate issues, then move to the explorer to inspect the underlying rows.",
            },
            {
                "eyebrow": "Output",
                "title": "The explorer doubles as an appendix builder",
                "body": "Narrow the table, sort it, and export the exact slice needed for review or reporting.",
            },
        ]
    )

    anomaly_tab, explorer_tab = st.tabs(["Anomaly detection", "Data explorer"])

    with anomaly_tab:
        render_section_intro(
            "Rolling Surprise Detection",
            "This workflow highlights local departures from recent history. It is useful for investigation triage, but the flags still need process context before they mean anything operationally.",
        )
        render_help_tip("A larger rolling window gives a slower-moving baseline. A higher z-score threshold shows only more extreme departures.")
        candidates = available_columns(master_df, [NH3, H2S, RAW_NH3, RAW_H2S, TEMP_NH3, TEMP_H2S, "total_gpm", "lbs_per_min", "nh3_per_lb", "h2s_per_lb"])
        with st.form("diag_anomaly_form"):
            target_col = st.selectbox("Signal", candidates)
            window = st.slider("Rolling window (minutes)", min_value=60, max_value=4320, value=1440, step=60)
            threshold = st.slider("Absolute z-score threshold", min_value=2.0, max_value=6.0, value=3.0, step=0.25)
            st.form_submit_button("Update anomaly detection")
        st.caption(
            "Higher thresholds show only more extreme departures. "
            "Lower thresholds are more sensitive but will usually produce more candidate anomalies to review."
        )

        z = add_zscore(master_df, target_col, window=window)
        anomalies = detect_anomalies(master_df, target_col, threshold=threshold, window=window)
        st.metric("Anomalies found", f"{len(anomalies):,}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=master_df.index, y=master_df[target_col], mode="lines", name=display_label(target_col)))
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies[target_col], mode="markers", name="Anomalies", marker=dict(size=7)))
        fig.update_layout(
            title=f"Anomalies in {display_label(target_col)}",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h"),
            xaxis_title="Date / time",
            yaxis_title=display_label(target_col),
        )
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True, key="diag_anomaly_signal_chart")

        z_df = pd.DataFrame({target_col: master_df[target_col], "z_score": z}).dropna().tail(5000)
        z_fig = go.Figure()
        z_fig.add_trace(go.Scatter(x=z_df.index, y=z_df["z_score"], mode="lines", name="z-score"))
        z_fig.add_hline(y=threshold, line_dash="dash")
        z_fig.add_hline(y=-threshold, line_dash="dash")
        z_fig.update_layout(title=f"Rolling z-score for {display_label(target_col)}", template="plotly_white", hovermode="x unified", xaxis_title="Date / time", yaxis_title="z-score")
        st.plotly_chart(z_fig, use_container_width=True, key="diag_anomaly_zscore_chart")

        with st.expander("View anomaly rows", expanded=False):
            st.dataframe(anomalies.head(500), use_container_width=True, height=280)

    with explorer_tab:
        render_section_intro(
            "Filtered Data Explorer",
            "Use the table view when a chart raises a question and you need to inspect underlying rows, sort extremes, or export a supporting appendix.",
        )
        render_help_tip("Use sorting to bring the highest values to the top, then narrow columns to only the fields needed for review or export.")
        with st.form("diag_explorer_form"):
            dataset_name = st.selectbox(
                "Dataset",
                ["master_1min", "master_1h", "master_daily", "monthly_summary", "weekday_summary", "event_metrics"],
                key="diag_dataset",
            )
            st.form_submit_button("Load dataset")
        dataset_map = {
            "master_1min": master_df,
            "master_1h": hourly_df,
            "master_daily": daily_df,
            "monthly_summary": monthly_df,
            "weekday_summary": weekday_df,
            "event_metrics": event_metrics_df,
        }
        view_df = dataset_map[dataset_name]
        if view_df is None:
            st.info(f"{dataset_name} is not available.")
        else:
            numeric_candidates = [c for c in view_df.columns if pd.api.types.is_numeric_dtype(view_df[c])]
            with st.form("diag_explorer_controls_form"):
                show_cols = st.multiselect("Columns", list(view_df.columns), default=list(view_df.columns[: min(12, len(view_df.columns))]), key="diag_show_cols")
                if not show_cols:
                    show_cols = list(view_df.columns)
                max_rows = st.slider("Rows", min_value=20, max_value=2000, value=200, step=20, key="diag_rows")
                sort_col = st.selectbox("Sort by", [None] + list(view_df.columns), index=0, key="diag_sort")
                st.form_submit_button("Update explorer")
            display_df = view_df.copy()
            if sort_col is not None:
                try:
                    display_df = display_df.sort_values(sort_col, ascending=False)
                except Exception:
                    pass
            preview_df = display_df[show_cols].head(max_rows)
            st.download_button(
                "Download selection as CSV",
                preview_df.to_csv().encode("utf-8"),
                file_name=f"{dataset_name}_{start_ts.date()}_{end_ts.date()}.csv",
                mime="text/csv",
            )
            st.dataframe(preview_df, use_container_width=True, height=420)
            if len(numeric_candidates) >= 1:
                dist_col = st.selectbox("Numeric column", numeric_candidates, key="diag_dist_col")
                hist = go.Figure(data=[go.Histogram(x=view_df[dist_col].dropna(), nbinsx=50)])
                hist.update_layout(title=f"Distribution of {display_label(dist_col)}", template="plotly_white", xaxis_title=display_label(dist_col), yaxis_title="Count")
                st.plotly_chart(hist, use_container_width=True, key="diag_distribution_histogram")


def render_page(page, ctx):
    if page == "Executive Brief":
        render_executive_brief_page(ctx)
    elif page == "Operations Review":
        render_operations_review_page(ctx)
    elif page == "Chemistry & Dosing":
        render_chemistry_dosing_page(ctx)
    elif page == "Research Progress":
        render_research_progress_page(ctx)
    elif page == "Performance & Coverage":
        render_performance_coverage_page(ctx)
    elif page == "Diagnostics & Data":
        render_diagnostics_data_page(ctx)
