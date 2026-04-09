# explore.py

# NOTE:
# This script is exploratory only and not used in the final modeling pipeline.

import sys
from pathlib import Path

from config import PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from scripts.load_data import load_all_data
from scripts.preprocess import preprocess_data
from scripts.features import build_features


# --------------------------------------------------
# Plot output directory
# --------------------------------------------------
PLOT_DIR = Path("plots/explore")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def find_interesting_window(df, target, threshold=1.0, window_days=2):
    """
    Automatically find a time window where target exceeds threshold.
    """
    mask = df[target] > threshold
    if not mask.any():
        return None, None

    start = df.loc[mask].index.min()
    end = start + pd.Timedelta(days=window_days)
    return start, end


def plot_time_window(df, cols, start, end, title, filename):
    """
    Plot selected columns over a time window and save figure.
    """
    subset = df.loc[start:end, cols]

    plt.figure(figsize=(14, 5))
    subset.plot(ax=plt.gca())
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150)
    plt.close()


def plot_lag_relationship(df, target, lag_col, filename):
    """
    Scatter plot of lagged feature vs target.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(df[lag_col], df[target], s=4, alpha=0.3)
    plt.xlabel(lag_col)
    plt.ylabel(target)
    plt.title(f"{target} vs {lag_col}")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150)
    plt.close()


def correlation_summary(df, derived_features, target):
    """
    Return correlation of derived features with target.
    """
    corr = (
        df[derived_features + [target]]
        .corr()[target]
        .drop(target)
        .sort_values(key=abs, ascending=False)
    )
    return corr


if __name__ == "__main__":

    # --------------------------------------------------
    # Load pipeline
    # --------------------------------------------------
    raw = load_all_data()
    clean = preprocess_data(raw)
    df, targets, derived_features = build_features(clean)

    # --------------------------------------------------
    # Auto-select interesting windows
    # --------------------------------------------------
    nh3_start, nh3_end = find_interesting_window(df, "nh3_nh3_ppm", threshold=1.0)
    h2s_start, h2s_end = find_interesting_window(df, "h2s_h2s_ppm", threshold = df["h2s_h2s_ppm"].quantile(0.99))

    # --------------------------------------------------
    # NH3 time-series diagnostics
    # --------------------------------------------------
    if nh3_start is not None:
        plot_time_window(
            df,
            cols=[
                "nh3_nh3_ppm",
                "nh3_roll_mean_15min",
                "nh3_roll_max_15min"
            ],
            start=nh3_start,
            end=nh3_end,
            title="NH3: Raw vs Rolling (15 min)",
            filename="nh3_raw_vs_rolling_15min.png"
        )

        plot_time_window(
            df,
            cols=[
                "east_sludge_out_gpm_combined",
                "nh3_nh3_ppm"
            ],
            start=nh3_start,
            end=nh3_end,
            title="East Sludge Flow vs NH3",
            filename="east_sludge_flow_vs_nh3.png"
        )

        plot_lag_relationship(
            df,
            target="nh3_nh3_ppm",
            lag_col="nh3_lag_30min",
            filename="nh3_vs_lag30min.png"
        )

    # --------------------------------------------------
    # H2S time-series diagnostics
    # --------------------------------------------------
    if h2s_start is not None:
        plot_time_window(
            df,
            cols=[
                "h2s_h2s_ppm",
                "h2s_roll_max_15min"
            ],
            start=h2s_start,
            end=h2s_end,
            title="H2S: Raw vs Rolling Max (15 min)",
            filename="h2s_raw_vs_rollingmax_15min.png"
        )

        plot_lag_relationship(
            df,
            target="h2s_h2s_ppm",
            lag_col="h2s_lag_30min",
            filename="h2s_vs_lag30min.png"
        )

    # --------------------------------------------------
    # Correlation summaries
    # --------------------------------------------------
    print("\nTop correlations with NH3:")
    print(
        correlation_summary(
            df,
            derived_features,
            target="nh3_nh3_ppm"
        ).head(10)
    )

    print("\nTop correlations with H2S:")
    print(
        correlation_summary(
            df,
            derived_features,
            target="h2s_h2s_ppm"
        ).head(10)
    )
