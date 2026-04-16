from scripts.plotting import dual_axis_figure
from scripts.plotting import event_window_figure
from scripts.plotting import heatmap_matrix


def dual_axis(df, x=None, y1=None, y2=None, y1_label=None, y2_label=None, title=""):
    return dual_axis_figure(
        df,
        y1,
        y2,
        y1_label or y1,
        y2_label or y2,
        title,
        x_col=x,
        x_title="Time",
        margin=dict(l=40, r=40, t=60, b=40),
    )


def event_window_plot(df, x=None, y1=None, y2=None, title=""):
    plot_df = df.copy()
    if x is not None and x != "minutes_from_event":
        plot_df["minutes_from_event"] = plot_df[x]
    return event_window_figure(
        plot_df,
        y1,
        y2,
        y1,
        y2,
        title,
    )


def heatmap(matrix, title=""):
    return heatmap_matrix(matrix, title=title)
