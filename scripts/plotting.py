import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_LAYOUT = {
    "template": "plotly_white",
    "hovermode": "x unified",
    "legend": dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        bgcolor="rgba(255,255,255,0.72)",
        bordercolor="rgba(24,35,31,0.08)",
        borderwidth=1,
        font=dict(size=11),
    ),
    "paper_bgcolor": "rgba(255,255,255,0)",
    "plot_bgcolor": "#fcfaf5",
    "font": dict(color="#18231f", family="IBM Plex Sans, Arial, sans-serif"),
}

PRIMARY_COLOR = "#1f6a53"
SECONDARY_COLOR = "#8b5e1a"
TERTIARY_COLOR = "#4b645b"
GRID_COLOR = "rgba(24,35,31,0.08)"
EVENT_LINE_COLOR = "rgba(31,106,83,0.28)"
PLANT_EVENT_COLOR = "#8f3f2b"


def apply_executive_axes(fig):
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor="rgba(24,35,31,0.18)",
        tickfont=dict(size=11),
        title_font=dict(size=12),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor="rgba(24,35,31,0.18)",
        tickfont=dict(size=11),
        title_font=dict(size=12),
    )
    fig.update_layout(title=dict(font=dict(size=20, color="#18231f"), x=0.01, xanchor="left"))
    return fig


def has_data(df, col):
    return df is not None and col in df.columns and df[col].notna().any()


def add_event_lines_plotly(fig, events, yref="paper", include_labels=True, plant_events=None):
    for name, times in events.items():
        for t in times:
            fig.add_vline(x=t, line_dash="dash", line_width=1, line_color=EVENT_LINE_COLOR, opacity=0.8)
            if include_labels:
                fig.add_annotation(
                    x=t,
                    y=1,
                    yref=yref,
                    text=name,
                    textangle=-90,
                    showarrow=False,
                    xanchor="left",
                    yanchor="top",
                    font=dict(size=8, color="#48665b"),
                    bgcolor="rgba(252,250,245,0.78)",
                )

    if not plant_events:
        return fig

    for name, t in plant_events.items():
        fig.add_vline(x=t, line_dash="dot", line_color=PLANT_EVENT_COLOR, line_width=1.5)
        if include_labels:
            fig.add_annotation(
                x=t,
                y=1,
                yref=yref,
                text=name,
                textangle=-90,
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(size=9, color=PLANT_EVENT_COLOR),
                bgcolor="rgba(252,250,245,0.84)",
            )

    return fig


def dual_axis_figure(
    df,
    y1_col,
    y2_col,
    y1_label,
    y2_label,
    title,
    *,
    x_col=None,
    x_title="Date / Time",
    add_events=None,
    plant_events=None,
    bar_second=False,
    customdata=None,
    y1_hover_prefix=None,
    y2_hover_prefix=None,
    rangeslider=True,
    margin=None,
    secondary_tickformat=None,
):
    fig = go.Figure()

    if not has_data(df, y1_col):
        return fig

    x_vals = df.index if x_col is None else df[x_col]
    cols = [c for c in [y1_col, y2_col] if c in df.columns]
    plot_df = df[cols].copy()
    plot_df["_x"] = x_vals
    if customdata is not None:
        plot_df["_customdata"] = customdata
    plot_df = plot_df.dropna(how="all", subset=cols)

    if plot_df.empty:
        return fig

    y1_hover = y1_hover_prefix or "Time: %{x}<br>"
    fig.add_trace(
        go.Scatter(
            x=plot_df["_x"],
            y=plot_df[y1_col],
            mode="lines",
            name=y1_label,
            line=dict(width=2.6, color=PRIMARY_COLOR),
            yaxis="y",
            customdata=plot_df["_customdata"] if "_customdata" in plot_df.columns else None,
            hovertemplate=y1_hover + f"{y1_label}: " + "%{y:.2f}<extra></extra>",
        )
    )

    if y2_col in plot_df.columns:
        y2_hover = y2_hover_prefix or "Time: %{x}<br>"
        if bar_second:
            fig.add_trace(
                go.Bar(
                    x=plot_df["_x"],
                    y=plot_df[y2_col],
                    name=y2_label,
                    yaxis="y2",
                    marker=dict(color="rgba(139, 94, 26, 0.55)", line=dict(color=SECONDARY_COLOR, width=0.8)),
                    hovertemplate=y2_hover + f"{y2_label}: " + "%{y:.2f}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["_x"],
                    y=plot_df[y2_col],
                    mode="lines",
                    name=y2_label,
                    line=dict(width=2, dash="dot", color=SECONDARY_COLOR),
                    yaxis="y2",
                    hovertemplate=y2_hover + f"{y2_label}: " + "%{y:.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        title=title,
        xaxis=dict(title=x_title),
        yaxis=dict(title=y1_label),
        yaxis2=dict(
            title=y2_label,
            overlaying="y",
            side="right",
            tickformat=secondary_tickformat,
        ),
        barmode="overlay" if bar_second else None,
        margin=margin or dict(l=100, r=100, t=100, b=100),
        **DEFAULT_LAYOUT,
    )
    apply_executive_axes(fig)

    if add_events:
        add_event_lines_plotly(fig, add_events, plant_events=plant_events)
    if rangeslider:
        fig.update_xaxes(rangeslider_visible=True)
    return fig


def event_window_figure(window_df, y1, y2, y1_label, y2_label, title, *, bar=False):
    fig = dual_axis_figure(
        window_df,
        y1,
        y2,
        y1_label,
        y2_label,
        title,
        x_col="minutes_from_event",
        x_title="Minutes from Event",
        bar_second=bar,
        customdata=window_df.index,
        y1_hover_prefix="Time: %{customdata}<br>",
        y2_hover_prefix="",
        rangeslider=False,
    )
    fig.add_vline(x=0, line_dash="dash", line_color=TERTIARY_COLOR)
    return fig


def event_study_figure(summary, title, ylabel):
    fig = go.Figure()
    if summary is None or summary.empty:
        return fig

    fig.add_trace(go.Scatter(x=summary.index, y=summary["q75"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=summary.index,
            y=summary["q25"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(31,106,83,0.18)",
            line=dict(width=0),
            name="IQR (25–75%)",
            hovertemplate="Q25: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary.index,
            y=summary["median"],
            mode="lines",
            name="Median",
            line=dict(width=3, color=PRIMARY_COLOR),
            hovertemplate="Median: %{y:.2f}<br>Δmin: %{x}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color=SECONDARY_COLOR, annotation_text="Event", annotation_position="top")
    fig.update_layout(
        title=title,
        xaxis_title="Minutes from event",
        yaxis_title=ylabel,
        margin=dict(l=100, r=100, t=100, b=100),
        **DEFAULT_LAYOUT,
    )
    apply_executive_axes(fig)
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def correlation_heatmap(df, cols, title="Correlation Heatmap"):
    corr = df[cols].corr(numeric_only=True)
    labels = [str(col).replace("_", " ") for col in corr.columns]
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[
                [0.0, "#7f3b08"],
                [0.18, "#b35806"],
                [0.5, "#f7f3ea"],
                [0.82, "#3e7f6a"],
                [1.0, "#1f6a53"],
            ],
            xgap=2,
            ygap=2,
            colorbar={
                "title": "Correlation",
                "tickformat": ".2f",
                "len": 0.8,
            },
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=max(520, 90 + 36 * len(labels)),
        margin=dict(l=100, r=100, t=100, b=100),
        xaxis_title="",
        yaxis_title="",
        **DEFAULT_LAYOUT,
    )
    fig.update_xaxes(side="bottom", tickangle=-35)
    fig.update_yaxes(autorange="reversed")

    for i, y_label in enumerate(labels):
        for j, x_label in enumerate(labels):
            value = corr.values[i, j]
            font_color = "#fffdf7" if abs(value) >= 0.45 else "#18231f"
            fig.add_annotation(
                x=x_label,
                y=y_label,
                text=f"{value:.2f}",
                showarrow=False,
                font=dict(size=11, color=font_color),
            )
    return fig


def heatmap_matrix(matrix, title=""):
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=list(matrix.columns),
            y=list(matrix.index),
            xgap=1,
            ygap=1,
            colorscale="YlGnBu",
            colorbar={"tickformat": ".2f"},
            hovertemplate="Value: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Column",
        yaxis_title="Row",
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
        **DEFAULT_LAYOUT,
    )

    values = np.asarray(matrix.values, dtype=float)
    vmax = np.nanmax(np.abs(values)) if values.size else 0
    threshold = 0.45 * vmax if vmax else 0
    for i, row_label in enumerate(matrix.index):
        for j, col_label in enumerate(matrix.columns):
            value = matrix.iloc[i, j]
            if pd.isna(value):
                continue
            font_color = "#fffdf7" if abs(value) >= threshold else "#18231f"
            fig.add_annotation(
                x=col_label,
                y=row_label,
                text=f"{value:.2f}",
                showarrow=False,
                font=dict(size=11, color=font_color),
            )
    return fig


def scatter_with_trend(df, x_col, y_col, color_col=None, title=""):
    plot_df = df[[c for c in [x_col, y_col, color_col] if c and c in df.columns]].dropna().copy()
    fig = go.Figure()
    if plot_df.empty:
        return fig

    marker_kwargs = dict(size=6, opacity=0.55)
    if color_col and color_col in plot_df.columns:
        marker_kwargs["color"] = plot_df[color_col]
        marker_kwargs["colorscale"] = "Viridis"
        marker_kwargs["showscale"] = True
        marker_kwargs["colorbar"] = {"title": color_col}

    fig.add_trace(
        go.Scatter(
            x=plot_df[x_col],
            y=plot_df[y_col],
            mode="markers",
            marker=marker_kwargs,
            name="Observations",
            text=plot_df.index.astype(str) if isinstance(plot_df.index, pd.DatetimeIndex) else None,
            hovertemplate=f"{x_col}: %{{x:.3f}}<br>{y_col}: %{{y:.3f}}<extra></extra>",
        )
    )

    if len(plot_df) >= 2:
        x = plot_df[x_col].astype(float).to_numpy()
        y = plot_df[y_col].astype(float).to_numpy()
        if np.isfinite(x).all() and np.isfinite(y).all() and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=slope * xs + intercept,
                    mode="lines",
                    name=f"Trend (slope={slope:.3f})",
                    line=dict(color=SECONDARY_COLOR, width=2.5),
                )
            )

    fig.update_layout(
        title=title or f"{y_col} vs {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        margin=dict(l=100, r=100, t=100, b=100),
        **DEFAULT_LAYOUT,
    )
    apply_executive_axes(fig)
    return fig


def multi_panel_figure(df, events, y1, y2, y1_label, y2_label, title):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=list(events.keys()),
        specs=[[{"secondary_y": True}, {"secondary_y": True}], [{"secondary_y": True}, {"secondary_y": True}]],
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (event_name, window_df), (r, c) in zip(events.items(), positions):
        if window_df.empty or y1 not in window_df.columns or y2 not in window_df.columns:
            continue

        fig.add_trace(
            go.Scatter(
                x=window_df["minutes"],
                y=window_df[y1],
                mode="lines",
                name=y1_label,
                line=dict(width=2.4, color=PRIMARY_COLOR),
                showlegend=(r == 1 and c == 1),
                hovertemplate=f"{y1_label}: " + "%{y:.2f}<br>Δmin: %{x}<extra></extra>",
            ),
            row=r,
            col=c,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=window_df["minutes"],
                y=window_df[y2],
                mode="lines",
                name=y2_label,
                line=dict(dash="dot", color=SECONDARY_COLOR, width=2),
                showlegend=(r == 1 and c == 1),
                hovertemplate=f"{y2_label}: " + "%{y:.2f}<br>Δmin: %{x}<extra></extra>",
            ),
            row=r,
            col=c,
            secondary_y=True,
        )
        fig.add_vline(x=0, line_dash="dash", line_color=TERTIARY_COLOR, row=r, col=c)
        fig.update_yaxes(title_text=y1_label, row=r, col=c, secondary_y=False)
        fig.update_yaxes(title_text=y2_label, row=r, col=c, secondary_y=True)

    fig.update_layout(title=title, height=800, **DEFAULT_LAYOUT)
    fig.update_xaxes(title_text="Minutes from Event", rangeslider_visible=True)
    apply_executive_axes(fig)
    return fig
