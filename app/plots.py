import plotly.graph_objects as go


# --------------------------------------------------
# INTERNAL HELPERS
# --------------------------------------------------
def _get_x(df, x):
    """
    Resolves x-axis input:
    - None → use index
    - string → column
    - otherwise → assume already array-like
    """
    if x is None:
        return df.index
    if isinstance(x, str):
        if x not in df.columns:
            raise ValueError(f"Column '{x}' not found in DataFrame")
        return df[x]
    return x


def _validate_columns(df, *cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


# --------------------------------------------------
# DUAL AXIS PLOT
# --------------------------------------------------
def dual_axis(df, x=None, y1=None, y2=None,
              y1_label=None, y2_label=None, title=""):

    _validate_columns(df, y1, y2)
    x_vals = _get_x(df, x)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=df[y1],
        name=y1_label or y1,
        mode="lines",
        line=dict(width=2),
        hovertemplate=f"{y1_label or y1}: %{{y:.2f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=df[y2],
        name=y2_label or y2,
        mode="lines",
        yaxis="y2",
        line=dict(width=2, dash="dot"),
        hovertemplate=f"{y2_label or y2}: %{{y:.2f}}<extra></extra>"
    ))

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(title="Time"),
        yaxis=dict(title=y1_label or y1),
        yaxis2=dict(
            title=y2_label or y2,
            overlaying="y",
            side="right"
        ),
        legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


# --------------------------------------------------
# EVENT WINDOW PLOT
# --------------------------------------------------
def event_window_plot(df, x=None, y1=None, y2=None, title=""):

    _validate_columns(df, y1, y2)
    x_vals = _get_x(df, x)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=df[y1],
        name=y1,
        mode="lines",
        line=dict(width=2),
        hovertemplate=f"{y1}: %{{y:.2f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=df[y2],
        name=y2,
        mode="lines",
        yaxis="y2",
        line=dict(dash="dot"),
        hovertemplate=f"{y2}: %{{y:.2f}}<extra></extra>"
    ))

    # Event line at zero
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="black"
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(title="Minutes from Event"),
        yaxis=dict(title=y1),
        yaxis2=dict(
            title=y2,
            overlaying="y",
            side="right"
        ),
        legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


# --------------------------------------------------
# HEATMAP
# --------------------------------------------------
def heatmap(matrix, title=""):

    if matrix is None or matrix.empty:
        raise ValueError("Heatmap matrix is empty")

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=list(matrix.columns),
            y=list(matrix.index),
            hovertemplate="Value: %{z:.3f}<extra></extra>"
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Column",
        yaxis_title="Row",
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )

    return fig