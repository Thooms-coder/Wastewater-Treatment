import pandas as pd
import streamlit as st

from scripts.constants import H2S, NH3

from app.dashboard_ui import render_variable_glossary


def render_sidebar_header():
    st.sidebar.markdown(
        """
        <div style="padding:0.2rem 0 0.8rem 0;">
            <div style="font-size:0.72rem; letter-spacing:0.14em; text-transform:uppercase; font-weight:700; color:rgba(238,246,241,0.72);">
                Executive Reporting
            </div>
            <div style="font-size:1.55rem; font-weight:700; line-height:1.05; margin-top:0.2rem;">
                Wastewater Odor Analytics
            </div>
            <div style="font-size:0.92rem; color:rgba(238,246,241,0.78); margin-top:0.45rem; line-height:1.45;">
                Review plant odor performance, operating transitions, process load, and data confidence inside a single reporting window.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_selector(page_options):
    return st.sidebar.radio("Page", page_options)


def select_time_window(full_master_df):
    full_start = full_master_df.index.min().normalize()
    full_end = full_master_df.index.max().normalize()

    with st.sidebar.container():
        st.markdown("**Time Window**")
        st.caption(
            "The selected dates filter the entire dashboard. "
            "Timeline, event counts, event-study summaries, aggregates, anomalies, and explorer exports all update to this window."
        )
        window_mode = st.selectbox(
            "Range",
            ["Full record", "Last 30 days", "Last 60 days", "Custom"],
            index=0,
        )

        if window_mode == "Last 30 days":
            default_start = max(full_start, full_end - pd.Timedelta(days=29))
            default_end = full_end
        elif window_mode == "Last 60 days":
            default_start = max(full_start, full_end - pd.Timedelta(days=59))
            default_end = full_end
        else:
            default_start = full_start
            default_end = full_end

        selected_range = st.date_input(
            "Dates",
            value=(default_start.date(), default_end.date()),
            min_value=full_start.date(),
            max_value=full_end.date(),
        )

        if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
            start_date, end_date = selected_range
        else:
            start_date = selected_range
            end_date = selected_range

        if start_date > end_date:
            start_date, end_date = end_date, start_date

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

    return {
        "full_start": full_start,
        "full_end": full_end,
        "start_ts": start_ts,
        "end_ts": end_ts,
    }


def build_filtered_state(
    full_state,
    window_state,
    filter_time_indexed_df,
    build_period_summaries,
    compute_event_metrics_table,
):
    start_ts = window_state["start_ts"]
    end_ts = window_state["end_ts"]

    master_df = filter_time_indexed_df(full_state["master_df"], start_ts, end_ts)
    hourly_df = filter_time_indexed_df(full_state["hourly_df"], start_ts, end_ts)
    daily_df = filter_time_indexed_df(full_state["daily_df"], start_ts, end_ts)
    monthly_df, weekday_df = build_period_summaries(daily_df) if daily_df is not None else (full_state["monthly_df"], full_state["weekday_df"])
    events_table = full_state["events_table"][
        (full_state["events_table"]["timestamp"] >= start_ts) & (full_state["events_table"]["timestamp"] <= end_ts)
    ].reset_index(drop=True)
    all_events = {
        name: [ts for ts in times if start_ts <= ts <= end_ts]
        for name, times in full_state["all_events"].items()
    }
    event_metrics_df = compute_event_metrics_table(master_df) if master_df is not None else full_state["event_metrics_df"]

    return {
        "master_df": master_df,
        "hourly_df": hourly_df,
        "daily_df": daily_df,
        "monthly_df": monthly_df,
        "weekday_df": weekday_df,
        "events_table": events_table,
        "all_events": all_events,
        "event_metrics_df": event_metrics_df,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "struvite_obs_df": full_state["struvite_obs_df"],
        "chem_labs_df": full_state["chem_labs_df"],
    }


def render_sidebar_resources(master_df, hourly_df, daily_df, monthly_df, weekday_df, start_ts, end_ts, coverage_value):
    st.sidebar.markdown("**Sidebar tools**")
    status_tab, stats_tab, glossary_tab = st.sidebar.tabs(["Status", "Stats", "Guide"])

    with status_tab:
        st.write(f"1-min: {'✅' if master_df is not None else '❌'}")
        st.write(f"1-hour: {'✅' if hourly_df is not None else '❌'}")
        st.write(f"Daily: {'✅' if daily_df is not None else '❌'}")
        st.write(f"Monthly: {'✅' if monthly_df is not None else '❌'}")
        st.write(f"Weekday: {'✅' if weekday_df is not None else '❌'}")
        st.caption(f"Filtered window: {start_ts.date()} to {end_ts.date()}")
        st.caption(
            "A green check only means the dataset is available after filtering. "
            "It does not mean the selected window has strong gas-sensor coverage."
        )

    with stats_tab:
        st.write(f"Minute rows: {len(master_df):,}")
        st.write(f"NH3 coverage: {coverage_value(master_df, NH3, len(master_df))}")
        st.write(f"H2S coverage: {coverage_value(master_df, H2S, len(master_df))}")
        if daily_df is not None and not daily_df.empty:
            st.write(f"Daily rows: {len(daily_df):,}")

    with glossary_tab:
        st.caption("Plain-English definitions for the main dashboard variables and event-study metrics.")
        render_variable_glossary()
