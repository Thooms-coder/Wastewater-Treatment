from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from scripts.constants import H2S, NH3

try:
    from scripts.chemistry_features import (
        FERRIC_REDUCTION_DATE,
        FIXED_PH,
        mgL_from_lbs_per_day,
    )
except Exception:
    FERRIC_REDUCTION_DATE = pd.Timestamp("2026-01-07")
    FIXED_PH = 7.0

    def mgL_from_lbs_per_day(lbs_per_day: float, mgd: float) -> float:
        denom = float(mgd) * 8.34
        return float(lbs_per_day) / denom if denom != 0 else np.nan


APP_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #f1efe9;
    --panel: rgba(255, 255, 255, 0.58);
    --panel-strong: rgba(255, 255, 255, 0.78);
    --ink: #18231f;
    --muted: #5d6b66;
    --line: rgba(24, 35, 31, 0.1);
    --accent: #1f6a53;
    --accent-soft: rgba(31, 106, 83, 0.12);
    --warn: #8b5e1a;
    --shadow: 0 22px 48px rgba(24, 33, 30, 0.14);
    --shadow-soft: 0 12px 28px rgba(24, 33, 30, 0.08);
    --shadow-deep: 0 26px 60px rgba(18, 27, 24, 0.18);
    --highlight: rgba(255, 255, 255, 0.72);
    --radius: 20px;
}

html, body, [class*="css"]  {
    font-family: "IBM Plex Sans", sans-serif;
    color: var(--ink);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(31, 106, 83, 0.16), transparent 26%),
        radial-gradient(circle at top right, rgba(139, 94, 26, 0.12), transparent 22%),
        radial-gradient(circle at 50% 0%, rgba(255,255,255,0.52), transparent 28%),
        linear-gradient(180deg, #faf7f1 0%, var(--bg) 100%);
}

.block-container {
    max-width: 1320px;
    padding-top: 1.1rem;
    padding-bottom: 2.25rem;
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at top left, rgba(255,255,255,0.1), transparent 28%),
        linear-gradient(180deg, rgba(16, 39, 32, 0.98) 0%, rgba(28, 57, 47, 0.97) 58%, rgba(19, 41, 34, 0.98) 100%);
    border-right: 1px solid rgba(255,255,255,0.1);
    box-shadow: inset -1px 0 0 rgba(255,255,255,0.06);
}

[data-testid="stSidebar"] * {
    color: #eef6f1;
}

[data-testid="stSidebar"] .stCaption {
    color: rgba(238, 246, 241, 0.78);
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    background: rgba(255,255,255,0.04);
}

[data-testid="stSidebar"] [data-baseweb="tab-list"] {
    gap: 0.25rem;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 0.22rem;
}

[data-testid="stSidebar"] [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 0.34rem 0.55rem 0.38rem 0.55rem;
    color: rgba(244, 250, 246, 0.9);
    font-weight: 600;
}

[data-testid="stSidebar"] [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.88) 0%, rgba(235, 244, 239, 0.82) 100%);
    color: #16352b;
    border: 1px solid rgba(255,255,255,0.26);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.65), 0 8px 16px rgba(9, 22, 18, 0.12);
}

[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="base-input"] > div {
    background: rgba(255, 255, 255, 0.16);
    border-color: rgba(255, 255, 255, 0.24);
}

[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="base-input"] input,
[data-testid="stSidebar"] [data-baseweb="base-input"] div,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="base-input"] span {
    color: #f7fbf8 !important;
    -webkit-text-fill-color: #f7fbf8 !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] svg,
[data-testid="stSidebar"] [data-baseweb="base-input"] svg {
    fill: #f7fbf8;
}

[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: rgba(255, 255, 255, 0.14);
}

[data-testid="stSidebar"] input::placeholder {
    color: rgba(247, 251, 248, 0.84) !important;
    -webkit-text-fill-color: rgba(247, 251, 248, 0.84) !important;
}

[data-testid="stMetric"] {
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(180deg, rgba(255,255,255,0.82) 0%, rgba(247, 243, 236, 0.92) 100%);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 0.9rem 1rem;
    box-shadow: var(--shadow);
    backdrop-filter: blur(10px);
}

[data-testid="stMetric"]::before,
.app-hero::before,
.context-panel::before,
.section-intro::before,
.summary-card::before,
.executive-card::before,
.report-card::before {
    content: "";
    position: absolute;
    inset: 1px 1px auto 1px;
    height: 42%;
    border-radius: inherit;
    background: linear-gradient(180deg, rgba(255,255,255,0.62) 0%, rgba(255,255,255,0) 100%);
    pointer-events: none;
}

[data-testid="stDataFrame"], [data-testid="stPlotlyChart"], [data-testid="stExpander"] {
    border-radius: var(--radius);
}

[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"] {
    background:
        linear-gradient(180deg, rgba(255,255,255,0.72) 0%, rgba(248, 244, 237, 0.88) 100%);
    border: 1px solid var(--line);
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(10px);
}

div[data-testid="stMarkdownContainer"] p code {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.92em;
    background: rgba(24, 35, 31, 0.06);
    padding: 0.12rem 0.34rem;
    border-radius: 6px;
}

.app-shell {
    padding-bottom: 0.5rem;
}

.app-hero {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at top left, rgba(255,255,255,0.62), transparent 34%),
        linear-gradient(135deg, rgba(255,255,255,0.82) 0%, rgba(247, 243, 236, 0.92) 55%, rgba(236, 242, 238, 0.9) 100%);
    border: 1px solid rgba(255,255,255,0.45);
    border-radius: 26px;
    padding: 1.15rem 1.2rem 1rem 1.2rem;
    box-shadow: var(--shadow-deep);
    margin: 0.1rem 0 0.85rem 0;
    backdrop-filter: blur(12px);
}

.app-kicker {
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 0.35rem;
}

.app-title {
    font-size: 2rem;
    line-height: 1.04;
    font-weight: 700;
    margin: 0;
    color: var(--ink);
}

.app-subtitle {
    margin-top: 0.45rem;
    max-width: 78ch;
    font-size: 0.98rem;
    line-height: 1.48;
    color: var(--muted);
}

.page-note {
    background: linear-gradient(180deg, rgba(255, 251, 241, 0.98) 0%, rgba(253, 248, 238, 0.96) 100%);
    border: 1px solid rgba(139, 94, 26, 0.18);
    border-left: 6px solid var(--warn);
    border-radius: 18px;
    padding: 0.9rem 1rem;
    margin: 0.35rem 0 0.8rem 0;
    box-shadow: var(--shadow-soft);
}

.page-note-title {
    font-size: 0.86rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--warn);
    margin-bottom: 0.45rem;
}

.page-note p {
    color: #473923;
    line-height: 1.58;
    margin: 0.35rem 0;
}

.status-pill {
    display: inline-block;
    padding: 0.28rem 0.6rem;
    margin: 0.1rem 0.45rem 0.3rem 0;
    border-radius: 999px;
    background: var(--accent-soft);
    color: var(--accent);
    font-size: 0.8rem;
    font-weight: 600;
    border: 1px solid rgba(31, 106, 83, 0.14);
}

.context-band {
    display: grid;
    grid-template-columns: 1.3fr 1fr;
    gap: 1rem;
    margin: 0.15rem 0 0.85rem 0;
}

.context-panel, .section-intro {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at top left, rgba(255,255,255,0.46), transparent 36%),
        linear-gradient(180deg, rgba(255,255,255,0.76) 0%, rgba(248, 244, 237, 0.94) 100%);
    border: 1px solid rgba(255,255,255,0.4);
    border-radius: 20px;
    padding: 0.9rem 1rem;
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(10px);
}

.context-panel.executive {
    position: relative;
    background:
        radial-gradient(circle at top left, rgba(255,255,255,0.15), transparent 30%),
        linear-gradient(135deg, rgba(17, 48, 40, 0.98) 0%, rgba(29, 72, 58, 0.97) 60%, rgba(21, 58, 46, 0.98) 100%);
    border-color: rgba(255,255,255,0.12);
    color: #f3f7f4;
    box-shadow: var(--shadow-deep);
}

.context-panel.executive .context-label,
.context-panel.executive .context-title,
.context-panel.executive .context-copy {
    color: #f3f7f4;
}

.context-panel.executive .context-label {
    color: rgba(243, 247, 244, 0.84);
}

.context-label, .section-label {
    text-transform: uppercase;
    letter-spacing: 0.11em;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 0.35rem;
}

.context-title {
    font-size: 1.2rem;
    font-weight: 700;
    line-height: 1.15;
    margin-bottom: 0.28rem;
    color: var(--ink);
}

.context-copy, .section-copy {
    color: var(--muted);
    line-height: 1.55;
    font-size: 0.95rem;
    margin: 0;
}

.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 0.65rem;
}

.metric-pill {
    padding: 0.56rem 0.72rem;
    border-radius: 999px;
    background: rgba(31, 106, 83, 0.08);
    border: 1px solid rgba(31, 106, 83, 0.12);
    color: var(--ink);
    font-size: 0.84rem;
    line-height: 1.2;
}

.metric-pill strong {
    color: var(--accent);
    font-weight: 700;
    margin-right: 0.28rem;
}

.section-intro {
    margin: 0.15rem 0 0.65rem 0;
    border-left: 5px solid rgba(31, 106, 83, 0.18);
}

.section-title {
    font-size: 1.12rem;
    font-weight: 700;
    line-height: 1.18;
    color: var(--ink);
    margin-bottom: 0.2rem;
}

.section-copy {
    max-width: 82ch;
}

.table-caption {
    font-size: 0.84rem;
    color: var(--muted);
    margin-top: -0.15rem;
    margin-bottom: 0.45rem;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.8rem;
    margin: 0.2rem 0 0.95rem 0;
}

.summary-card {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at top left, rgba(255,255,255,0.58), transparent 36%),
        linear-gradient(180deg, rgba(255,255,255,0.82) 0%, rgba(247, 243, 236, 0.94) 100%);
    border: 1px solid rgba(255,255,255,0.42);
    border-radius: 18px;
    padding: 0.85rem 0.95rem 0.9rem 0.95rem;
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(9px);
    transform: translateY(0);
    transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
}

.summary-card:hover,
.executive-card:hover,
.report-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow);
    border-color: rgba(31, 106, 83, 0.18);
}

.summary-eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 0.35rem;
}

.summary-title {
    font-size: 1.02rem;
    line-height: 1.22;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 0.28rem;
}

.summary-body {
    font-size: 0.9rem;
    line-height: 1.48;
    color: var(--muted);
}

.summary-meta {
    margin-top: 0.48rem;
    font-size: 0.8rem;
    line-height: 1.35;
    color: var(--accent);
    font-weight: 600;
}

.executive-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.9rem;
    margin: 0.35rem 0 1rem 0;
}

.executive-card {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at top left, rgba(255,255,255,0.58), transparent 36%),
        linear-gradient(180deg, rgba(255,255,255,0.84) 0%, rgba(247, 243, 236, 0.95) 100%);
    border: 1px solid rgba(255,255,255,0.42);
    border-radius: 20px;
    padding: 0.9rem 0.95rem 0.85rem 0.95rem;
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(9px);
}

.executive-card-label {
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 0.45rem;
}

.executive-card-value {
    font-size: 1.55rem;
    line-height: 1;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 0.38rem;
}

.executive-card-note {
    font-size: 0.86rem;
    line-height: 1.4;
    color: var(--muted);
}

.brief-list {
    margin: 0.8rem 0 0 0;
    padding-left: 1.05rem;
    color: #f3f7f4;
}

.brief-list li {
    margin: 0.35rem 0;
    line-height: 1.45;
}

.report-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    background:
        radial-gradient(circle at top left, rgba(255,255,255,0.5), transparent 32%),
        linear-gradient(90deg, rgba(255,255,255,0.7) 0%, rgba(237, 244, 239, 0.88) 48%, rgba(248, 242, 232, 0.9) 100%);
    border: 1px solid rgba(255,255,255,0.42);
    border-radius: 18px;
    padding: 0.9rem 1rem;
    margin: 0.2rem 0 1rem 0;
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(9px);
}

.report-banner strong {
    display: block;
    color: var(--ink);
    margin-bottom: 0.1rem;
}

.report-banner span {
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.45;
}

.report-chip {
    white-space: nowrap;
    border-radius: 999px;
    padding: 0.45rem 0.7rem;
    background: rgba(255,255,255,0.94);
    border: 1px solid rgba(24, 35, 31, 0.12);
    color: #16352b;
    font-size: 0.82rem;
    font-weight: 700;
}

.report-two-col {
    display: grid;
    grid-template-columns: 1.45fr 1fr;
    gap: 1rem;
    margin: 0.25rem 0 1rem 0;
}

.report-card {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at top left, rgba(255,255,255,0.58), transparent 36%),
        linear-gradient(180deg, rgba(255,255,255,0.84) 0%, rgba(247, 243, 236, 0.95) 100%);
    border: 1px solid rgba(255,255,255,0.42);
    border-radius: 20px;
    padding: 0.95rem 1rem;
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(9px);
}

.report-card h3 {
    margin: 0 0 0.45rem 0;
    font-size: 1.05rem;
    color: var(--ink);
}

.report-card p, .report-card li {
    color: var(--muted);
    line-height: 1.5;
    font-size: 0.93rem;
}

.report-card ul {
    margin: 0.55rem 0 0 1rem;
    padding: 0;
}

.block-spacer {
    height: 0.35rem;
}

[data-testid="stRadio"] > div {
    gap: 0.5rem;
}

[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.72) 0%, rgba(244, 240, 233, 0.9) 100%);
    border: 1px solid rgba(255,255,255,0.42);
    border-radius: 999px;
    padding: 0.35rem 0.8rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.72), 0 6px 14px rgba(24, 33, 30, 0.05);
}

[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] {
    background: linear-gradient(180deg, rgba(224, 241, 235, 0.95) 0%, rgba(204, 232, 222, 0.92) 100%);
    border-color: rgba(31, 106, 83, 0.28);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.78), 0 10px 18px rgba(31, 106, 83, 0.12);
}

[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stDateInput"] label,
[data-testid="stSlider"] label,
[data-testid="stCheckbox"] label {
    font-weight: 600;
    color: var(--ink);
}

[data-testid="stAlert"] {
    border-radius: 16px;
}

[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.42);
    background:
        linear-gradient(180deg, rgba(255,255,255,0.72) 0%, rgba(247, 243, 236, 0.9) 100%);
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(8px);
}

[data-testid="stExpander"] details summary p {
    font-weight: 600;
    color: var(--ink);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.45rem;
    background: linear-gradient(180deg, rgba(255,255,255,0.7) 0%, rgba(245, 241, 235, 0.86) 100%);
    border: 1px solid rgba(255,255,255,0.42);
    border-radius: 999px;
    padding: 0.28rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.75);
}

.stTabs [data-baseweb="tab"] {
    height: auto;
    border-radius: 999px;
    padding: 0.42rem 0.85rem 0.46rem 0.85rem;
    color: #42534d;
    background: transparent;
    border: 1px solid transparent;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(224, 241, 235, 0.95) 0%, rgba(206, 232, 222, 0.92) 100%) !important;
    border-color: rgba(31, 106, 83, 0.18) !important;
    color: var(--accent) !important;
    box-shadow: 0 8px 16px rgba(31, 106, 83, 0.1), inset 0 1px 0 rgba(255,255,255,0.82);
}

[data-testid="stPopover"] button {
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.42);
    background: linear-gradient(180deg, rgba(255,255,255,0.72) 0%, rgba(230, 241, 236, 0.9) 100%);
    color: var(--accent);
    padding: 0.2rem 0.55rem;
    font-size: 0.84rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.8), 0 8px 16px rgba(24, 33, 30, 0.06);
}

[data-baseweb="select"] > div,
[data-baseweb="base-input"] > div {
    background: linear-gradient(180deg, rgba(255,255,255,0.74) 0%, rgba(245, 241, 235, 0.9) 100%);
    border: 1px solid rgba(255,255,255,0.42) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.78), 0 8px 18px rgba(24, 33, 30, 0.05);
}

[data-testid="stForm"] {
    background:
        radial-gradient(circle at top left, rgba(255,255,255,0.38), transparent 34%),
        linear-gradient(180deg, rgba(255,255,255,0.62) 0%, rgba(247, 243, 236, 0.86) 100%);
    border: 1px solid rgba(255,255,255,0.38);
    border-radius: 18px;
    padding: 0.85rem 0.9rem 0.55rem 0.9rem;
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(8px);
    margin-bottom: 0.7rem;
}

.stButton > button,
[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.36);
    background: linear-gradient(180deg, #2a7d62 0%, #1f6a53 52%, #184d3d 100%);
    color: #f6fbf8;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.22), 0 14px 24px rgba(31, 106, 83, 0.18);
    font-weight: 700;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {
    border-color: rgba(255,255,255,0.48);
    filter: brightness(1.03);
    transform: translateY(-1px);
}

@media (max-width: 960px) {
    .context-band {
        grid-template-columns: 1fr;
    }

    .executive-grid {
        grid-template-columns: 1fr;
    }

    .summary-grid {
        grid-template-columns: 1fr;
    }

    .report-two-col {
        grid-template-columns: 1fr;
    }
}

@media print {
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    [data-testid="stToolbar"],
    [data-testid="stHeader"],
    [data-testid="stRadio"],
    [data-testid="stSelectbox"],
    [data-testid="stMultiSelect"],
    [data-testid="stSlider"],
    [data-testid="stDateInput"],
    [data-testid="stCheckbox"],
    [data-testid="stDownloadButton"],
    button,
    .stTabs {
        display: none !important;
    }

    .stApp, .main, section.main > div {
        background: white !important;
    }

    .app-hero,
    .context-panel,
    .section-intro,
    .executive-card,
    .report-card,
    [data-testid="stMetric"] {
        box-shadow: none !important;
        break-inside: avoid;
    }
}
</style>
"""

OPTIONAL_TABLE_SCHEMAS = {
    "Struvite Observations": {
        "required_any": [
            {"timestamp", "date", "sample_date"},
            {"location", "sample_location", "asset"},
        ],
        "recommended": ["severity", "coupon_result", "notes"],
    },
    "Chemistry Lab Results": {
        "required_any": [
            {"timestamp", "date", "sample_date"},
            {"pH", "alkalinity", "mg", "nh4", "po4"},
        ],
        "recommended": ["sample_location", "notes"],
    },
}


def safe_delta(series):
    if series is None or len(series.dropna()) < 2:
        return None
    clean = series.dropna()
    return clean.iloc[-1] - clean.iloc[0]


def metric_value(df, col, fn="mean", fmt="{:.2f}"):
    if df is None or col not in df.columns or not df[col].notna().any():
        return "NA"
    series = df[col].dropna()
    value = getattr(series, fn)() if isinstance(fn, str) else fn(series)
    return fmt.format(value)


def coverage_value(df, col, expected_points):
    if df is None or col not in df.columns or not df[col].notna().any() or expected_points <= 0:
        return "NA"
    pct = (df[col].notna().sum() / expected_points) * 100
    return f"{pct:.1f}%"


def executive_summary(master_df, events_table):
    if master_df is None or master_df.empty:
        return []

    nh3_delta = safe_delta(master_df[NH3]) if NH3 in master_df.columns else None
    h2s_delta = safe_delta(master_df[H2S]) if H2S in master_df.columns else None
    avg_flow = metric_value(master_df, "total_gpm")
    transitions = len(events_table)
    nh3_cov = coverage_value(master_df, NH3, len(master_df))
    h2s_cov = coverage_value(master_df, H2S, len(master_df))

    messages = [
        f"The reporting window covers {master_df.index.min().date()} to {master_df.index.max().date()} with {transitions:,} detected chemical transitions.",
        f"Observed gas coverage is {nh3_cov} for NH3 and {h2s_cov} for H2S, which sets the confidence ceiling for every summary shown below.",
        f"Average plant flow during the window was {avg_flow} total GPM.",
    ]

    if nh3_delta is not None:
        direction = "up" if nh3_delta > 0 else "down" if nh3_delta < 0 else "flat"
        messages.append(f"NH3 finished the period {direction} by {abs(nh3_delta):.2f} ppm versus the opening level.")
    if h2s_delta is not None:
        direction = "up" if h2s_delta > 0 else "down" if h2s_delta < 0 else "flat"
        messages.append(f"H2S finished the period {direction} by {abs(h2s_delta):.2f} ppm versus the opening level.")

    return messages


def render_executive_cards(cards):
    if not cards:
        return

    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="executive-card">
                    <div class="executive-card-label">{escape(card["label"])}</div>
                    <div class="executive-card-value">{escape(card["value"])}</div>
                    <div class="executive-card-note">{escape(card["note"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_summary_cards(cards):
    if not cards:
        return

    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        eyebrow = escape(card.get("eyebrow", "Summary"))
        title = escape(card.get("title", ""))
        body = escape(card.get("body", ""))
        meta = escape(card.get("meta", ""))
        meta_markup = f'<div class="summary-meta">{meta}</div>' if meta else ""
        with column:
            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="summary-eyebrow">{eyebrow}</div>
                    <div class="summary-title">{title}</div>
                    <div class="summary-body">{body}</div>
                    {meta_markup}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_page_header(title, subtitle, kicker="Wastewater Treatment"):
    st.markdown(
        f"""
        <div class="app-shell">
            <div class="app-hero">
                <div class="app-kicker">{escape(kicker)}</div>
                <h1 class="app-title">{escape(title)}</h1>
                <div class="app-subtitle">{escape(subtitle)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_notes(page_name):
    notes = {
        "Executive Brief": """
        This page is the orientation layer for the dashboard.

        Use it to answer four questions first:
        1. What time window am I looking at?
        2. How much NH3 and H2S data is actually present in that window?
        3. Which chemical transitions occurred inside the window?
        4. Do the odor and operating signals appear to move together at a high level?

        Interpretation notes:
        - `NH3` is summarized from a 15-minute rolling average because ammonia usually behaves more like a sustained background condition than a single sharp burst.
        - `H2S` is summarized from a 15-minute rolling maximum because short sulfur spikes can be operationally important and noticeable even when the average value stays modest.
        - Coverage metrics matter. A low NH3 or H2S coverage percentage means the averages and event summaries may reflect only part of plant behavior in the selected period.
        - An `event` in this app means a detected chemistry state change at a specific timestamp, such as Ferric turning ON, Ferric turning OFF, HCl turning ON, or HCl turning OFF.
        - Event counts here are counts of those detected change timestamps inside the filtered window, not counts across the full project history.
        """,
        "Operations Review": """
        This page is for temporal pattern reading.

        Use the resolution selector based on the question:
        - `1-minute`: best for short spikes, event timing, and sensor behavior.
        - `1-hour`: best for relating odor to broader operating changes and hourly load context.
        - `Daily`: best for trend compression and longer-duration shifts.

        Reading guidance:
        - Dashed vertical lines are detected chemistry events: timestamps where Ferric or HCl changes state.
        - `ON` means the relevant chemistry flag switched from inactive to active. `OFF` means it switched from active to inactive.
        - Dotted purple lines are plant-level contextual events, such as the ferric reduction date.
        - When you overlay two signals, look for whether peaks, drops, or regime changes line up in time, not just whether the lines share a similar shape.
        - If a signal appears sparse or flat, check the selected window and data coverage before drawing a conclusion.
        """,
        "Chemistry & Dosing": """
        This page reconnects the app to the research chemistry questions.

        Use it to review:
        - whether ferric dose features are present in the current data,
        - what those dose features imply in lb/day and mg/L terms,
        - how odor signals appear to move relative to dose and flow context,
        - and what still needs to be added before the app can support struvite claims.

        Important note:
        - The current chemistry features are contextual and descriptive.
        - They do not yet replace bench chemistry experiments or equilibrium/speciation modeling.
        """,
        "Research Progress": """
        This page is the thesis-program tracker rather than a pure analysis page.

        Use it to monitor:
        - what is active in the bench-scale and full-scale work,
        - what has stalled or failed and needs documentation,
        - how methods logging is being maintained,
        - and how the thesis structure is being built in parallel with the experiments.

        Reading note:
        - The committee explicitly asked for failed methods, milestones, and writing progress to be documented as part of the project, not just successful plots and analyses.
        """,
        "Performance & Coverage": """
        This page compresses the filtered window into daily, monthly, weekday, and coverage views.

        What each tab is for:
        - `Daily`: inspect day-to-day movement in odor and operating load.
        - `Monthly`: compare broad monthly operating levels within the filtered selection.
        - `Weekday`: check whether weekday structure appears in the selected period.
        - `Coverage`: verify whether missingness could distort interpretation.

        Interpretation note:
        - These aggregates are recalculated from the filtered daily dataset, so changing the sidebar date window changes the summaries shown here.
        - If the filtered period is short, monthly and weekday summaries may be descriptive but not representative.
        """,
        "Diagnostics & Data": """
        This page highlights unusual points using a rolling z-score.

        How it works:
        - The app compares the current value to a rolling mean and rolling standard deviation over the selected window length.
        - Large absolute z-scores indicate points that are unusual relative to recent history.

        Practical guidance:
        - Use shorter rolling windows to detect local spikes.
        - Use longer rolling windows to find larger regime departures.
        - Treat anomaly flags as candidates for review, not confirmed bad data or confirmed process upsets.

        The second chart shows the z-score itself so you can see whether anomalies are isolated bursts or part of a sustained excursion.
        """,
    }

    if page_name in notes:
        chunks = [line.strip() for line in notes[page_name].strip().split("\n\n") if line.strip()]
        preview = chunks[0]
        with st.expander("How To Read This Page", expanded=False):
            st.markdown(
                f"""
                <div class="page-note">
                    <div class="page-note-title">Reading Guide</div>
                    <p>{escape(preview)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for block in chunks[1:]:
                st.markdown(block)


def render_context_band(start_ts, end_ts, rows, transitions, nh3_cov, h2s_cov):
    st.markdown(
        f"""
        <div class="context-band">
            <div class="context-panel">
                <div class="context-label">Study Window</div>
                <div class="context-title">{start_ts.date()} to {end_ts.date()}</div>
                <p class="context-copy">
                    Every chart, event count, aggregate, anomaly flag, and table on this page is scoped to the currently selected filter window.
                </p>
                <div class="pill-row">
                    <div class="metric-pill"><strong>Rows</strong>{rows:,}</div>
                    <div class="metric-pill"><strong>Transitions</strong>{transitions:,}</div>
                    <div class="metric-pill"><strong>NH3 coverage</strong>{nh3_cov}</div>
                    <div class="metric-pill"><strong>H2S coverage</strong>{h2s_cov}</div>
                </div>
            </div>
            <div class="context-panel">
                <div class="context-label">Reading Focus</div>
                <div class="context-title">Interpret the window before interpreting the signal.</div>
                <p class="context-copy">
                    Coverage, event density, and time scale all change what a pattern means. Use the filter window as part of the analysis, not just as a convenience control.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_executive_brief(master_df, events_table):
    bullets = "".join(f"<li>{escape(item)}</li>" for item in executive_summary(master_df, events_table))
    st.markdown(
        f"""
        <div class="context-panel executive">
            <div class="context-label">Executive Brief</div>
            <div class="context-title">What matters in this reporting window</div>
            <p class="context-copy">
                This summary is designed for briefing and review. It compresses the current window into a short narrative before the detailed charts.
            </p>
            <ul class="brief-list">{bullets}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_report_banner():
    st.markdown(
        """
        <div class="report-banner">
            <div>
                <strong>Printable Report Layout</strong>
                <span>
                    Use the browser print dialog on this page for a clean reporting export. Controls and navigation are suppressed in print view.
                </span>
            </div>
            <div class="report-chip">Optimized For PDF Export</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_report_highlights(master_df, events_table, event_metrics_df):
    effect_lines = []
    if event_metrics_df is not None and not event_metrics_df.empty:
        ranked = event_metrics_df.copy()
        if "percent_change" in ranked.columns:
            ranked = ranked.assign(abs_change=ranked["percent_change"].abs()).sort_values("abs_change", ascending=False)
        ranked = ranked.head(3)
        for _, row in ranked.iterrows():
            if {"chemical", "event_type", "signal", "percent_change"}.issubset(ranked.columns):
                effect_lines.append(
                    f"{row['chemical']} {row['event_type']} on {row['signal']}: {row['percent_change']:.1f}% median change."
                )

    if not effect_lines:
        effect_lines = [
            "Event effect metrics are not available for this reporting window.",
            "Use the transition tables below when a narrative explanation is still required.",
        ]

    summary_lines = executive_summary(master_df, events_table)[:3]
    summary_markup = "".join(f"<li>{escape(item)}</li>" for item in summary_lines)
    effect_markup = "".join(f"<li>{escape(item)}</li>" for item in effect_lines)

    st.markdown(
        f"""
        <div class="report-two-col">
            <div class="report-card">
                <h3>Management Summary</h3>
                <ul>{summary_markup}</ul>
            </div>
            <div class="report-card">
                <h3>Largest Estimated Effects</h3>
                <ul>{effect_markup}</ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_research_alignment_df():
    return pd.DataFrame(
        [
            {
                "objective": "1. Explain chemical mechanisms for NH3/H2S reduction",
                "current_repo_support": "Partial",
                "where_to_review": "Chemistry & Dosing page, fixed chemistry features, event-response views",
                "current_status": "Fixed chemistry context and ferric dose fields exist, but mechanistic interpretation is still descriptive rather than validated chemistry modeling.",
            },
            {
                "objective": "2. Optimize FeCl3 and HCl for odor mitigation",
                "current_repo_support": "Partial",
                "where_to_review": "Operations Review and Chemistry & Dosing pages",
                "current_status": "The app supports odor/event analysis well. Ferric dosing is partly represented; HCl dosage optimization is not yet data-backed in the repo.",
            },
            {
                "objective": "3. Minimize struvite formation",
                "current_repo_support": "Placeholder",
                "where_to_review": "Chemistry & Dosing page: struvite workflow tracker",
                "current_status": "No direct struvite measurement table is in the app yet. The workflow placeholder makes the gap explicit and defines the needed future inputs.",
            },
            {
                "objective": "4. Validate full-scale implementation with continuous monitors",
                "current_repo_support": "Strong",
                "where_to_review": "Executive Brief, Operations Review, Performance & Coverage, Diagnostics & Data",
                "current_status": "This is the strongest part of the repo today: continuous odor monitoring, transition detection, coverage analysis, and reporting views are in place.",
            },
        ]
    )


def build_research_progress_df():
    return pd.DataFrame(
        [
            {
                "lane": "Bench-scale H2S method",
                "current_status": "In development",
                "evidence": "Committee discussion indicates earlier approaches failed and a new liquid-phase standard or meter setup is being used.",
                "next_milestone": "Run ferric-vs-H2S concentration series and identify the plateau or failure point.",
            },
            {
                "lane": "Bench-scale NH3/HCl method",
                "current_status": "Not yet producing usable results",
                "evidence": "Meeting discussion indicates no thesis-ready bench NH3 dataset is available yet.",
                "next_milestone": "Stand up a repeatable NH3/HCl bench method and log conditions and outcomes.",
            },
            {
                "lane": "Full-scale odor analytics",
                "current_status": "Active",
                "evidence": "Continuous NH3/H2S, flow, event, and chemistry context are integrated in the current dashboard and pipeline.",
                "next_milestone": "Absorb new plant data and extend optimization analysis around mg/L dose and operating conditions.",
            },
            {
                "lane": "Struvite and scaling outcomes",
                "current_status": "Partial observational only",
                "evidence": "The committee described ongoing plant observations, but no structured struvite/scaling dataset exists in the repo yet.",
                "next_milestone": "Start a structured observation log linking scaling, pH, dose, flow, and odor conditions.",
            },
            {
                "lane": "Thesis writing and methods",
                "current_status": "Needs active drafting",
                "evidence": "Committee asked for methods, failed paths, introduction, and milestones to be written in parallel with experiments.",
                "next_milestone": "Maintain methods log and thesis outline/status document as living artifacts.",
            },
        ]
    )


def build_methods_log_template_df():
    return pd.DataFrame(
        [
            {
                "date": "2026-04-19",
                "lane": "Bench-scale H2S method",
                "experiment_or_method": "Liquid-phase H2S standard with ferric titration setup",
                "status": "Planned",
                "what_worked": "",
                "what_failed": "",
                "notes": "Document standard concentration, meter setup, temperature, and each ferric increment.",
                "next_step": "Run first concentration series and record plateau behavior.",
            },
            {
                "date": "2026-04-19",
                "lane": "Full-scale analytics",
                "experiment_or_method": "Dashboard mg/L dose and event-study workflow",
                "status": "Active",
                "what_worked": "Full-scale NH3/H2S and flow context can be visualized and tested.",
                "what_failed": "",
                "notes": "Log each new plant data drop and what changed in the analysis.",
                "next_step": "Ingest the next available full-scale dataset.",
            },
        ]
    )


def build_thesis_outline_df():
    return pd.DataFrame(
        [
            {"section": "1. Introduction and problem framing", "current_state": "Should begin now", "repo_anchor": "notes/thesis_outline_status.md"},
            {"section": "2. Literature review and wastewater process context", "current_state": "Should begin now", "repo_anchor": "notes/thesis_outline_status.md"},
            {"section": "3. Bench-scale H2S methods and results", "current_state": "Methods in development; results incomplete", "repo_anchor": "notes/methods_log_template.csv"},
            {"section": "4. Bench-scale NH3/HCl methods and results", "current_state": "Not yet mature", "repo_anchor": "notes/methods_log_template.csv"},
            {"section": "5. Full-scale Dayton analysis", "current_state": "Strongest current evidence base", "repo_anchor": "app/page_renderers.py"},
            {"section": "6. Struvite and scaling outcomes", "current_state": "Partial observational basis only", "repo_anchor": "notes/thesis_outline_status.md"},
            {"section": "7. Discussion, optimization, and recommendations", "current_state": "Depends on integrating bench and full-scale results", "repo_anchor": "notes/thesis_outline_status.md"},
        ]
    )


def render_research_alignment():
    render_section_intro(
        "Research Objective Alignment",
        "This table maps the current app and repo to the dissertation objectives described in the notes, so progress can be assessed against the research plan rather than the UI alone.",
    )
    render_help_tip("Use this as the project scorecard. A polished dashboard does not mean every research objective is equally mature.")
    st.dataframe(build_research_alignment_df(), use_container_width=True, height=260)


def compute_ferric_mgL_series(df):
    if df is None or df.empty:
        return pd.Series(dtype=float, name="ferric_active_mg_per_L")
    required = {"ferric_active_lbs_per_day", "total_gpm"}
    if not required.issubset(df.columns):
        return pd.Series(dtype=float, index=df.index, name="ferric_active_mg_per_L")

    mgd = df["total_gpm"] * 1440 / 1_000_000
    mgl = [
        mgL_from_lbs_per_day(lbs, flow_mgd)
        if pd.notna(lbs) and pd.notna(flow_mgd) and flow_mgd > 0
        else np.nan
        for lbs, flow_mgd in zip(df["ferric_active_lbs_per_day"], mgd)
    ]
    return pd.Series(mgl, index=df.index, name="ferric_active_mg_per_L")


def compute_hcl_mgL_series(df):
    if df is None or df.empty:
        return pd.Series(dtype=float, name="hcl_active_mg_per_L")
    required = {"hcl_active_lbs_per_day", "total_gpm"}
    if not required.issubset(df.columns):
        return pd.Series(dtype=float, index=df.index, name="hcl_active_mg_per_L")

    mgd = df["total_gpm"] * 1440 / 1_000_000
    mgl = [
        mgL_from_lbs_per_day(lbs, flow_mgd)
        if pd.notna(lbs) and pd.notna(flow_mgd) and flow_mgd > 0
        else np.nan
        for lbs, flow_mgd in zip(df["hcl_active_lbs_per_day"], mgd)
    ]
    return pd.Series(mgl, index=df.index, name="hcl_active_mg_per_L")


def build_chemistry_review_table(df):
    rows = [
        {
            "workflow_element": "Fixed chemistry context",
            "status": "Available",
            "current_value": f"pH_fixed = {FIXED_PH:.1f}",
            "notes": "Static chemistry context is available for interpretation, not full equilibrium modeling.",
        },
        {
            "workflow_element": "Ferric reduction date",
            "status": "Available",
            "current_value": str(FERRIC_REDUCTION_DATE.date()),
            "notes": "Used in ferric dose feature logic to halve representative dosing after the plant change.",
        },
    ]

    if df is not None and "ferric_solution_lbs_per_day" in df.columns:
        sol = df["ferric_solution_lbs_per_day"].dropna()
        rows.append(
            {
                "workflow_element": "Ferric solution dose",
                "status": "Available" if not sol.empty else "Missing in current window",
                "current_value": f"{sol.median():.1f} lb/day median" if not sol.empty else "NA",
                "notes": "Derived from ferric availability assumptions and reduction date logic.",
            }
        )

    if df is not None and "hcl_solution_lbs_per_day" in df.columns:
        sol = df["hcl_solution_lbs_per_day"].dropna()
        rows.append(
            {
                "workflow_element": "HCl solution dose",
                "status": "Available" if not sol.empty else "Missing in current window",
                "current_value": f"{sol.median():.1f} lb/day median" if not sol.empty else "NA",
                "notes": "Derived from HCl availability assumptions and should be replaced by measured feed data when available.",
            }
        )

    ferric_mgl = compute_ferric_mgL_series(df)
    ferric_mgl_clean = ferric_mgl.dropna()
    rows.append(
        {
            "workflow_element": "Ferric active dose intensity",
            "status": "Available" if not ferric_mgl_clean.empty else "Missing in current window",
            "current_value": f"{ferric_mgl_clean.median():.2f} mg/L median" if not ferric_mgl_clean.empty else "NA",
            "notes": "Converted from active lb/day using flow-derived MGD and the standard 8.34 wastewater conversion.",
        }
    )

    hcl_mgl = compute_hcl_mgL_series(df)
    hcl_mgl_clean = hcl_mgl.dropna()
    rows.append(
        {
            "workflow_element": "HCl active dose intensity",
            "status": "Available" if not hcl_mgl_clean.empty else "Missing in current window",
            "current_value": f"{hcl_mgl_clean.median():.2f} mg/L median" if not hcl_mgl_clean.empty else "NA",
            "notes": "Converted from active lb/day using flow-derived MGD and the standard 8.34 wastewater conversion.",
        }
    )

    hcl_candidates = [c for c in ["hcl_solution_lbs_per_day", "hcl_active_lbs_per_day", "hcl_active_mg_per_L"] if df is not None and c in df.columns]
    rows.append(
        {
            "workflow_element": "HCl dosing features",
            "status": "Available" if hcl_candidates else "Not yet implemented",
            "current_value": ", ".join(hcl_candidates) if hcl_candidates else "No HCl dose columns in current repo outputs",
            "notes": "The notes make HCl optimization central, so measured feed data should replace representative assumptions when available.",
        }
    )

    return pd.DataFrame(rows)


def build_struvite_placeholder_df():
    return pd.DataFrame(
        [
            {
                "needed_input": "Scale observation log or coupon inspection table",
                "current_status": "Missing",
                "planned_metric": "Presence/absence, severity class, or % reduction versus baseline",
                "why_it_matters": "Needed to connect odor-control chemistry to struvite control claims in the research notes.",
            },
            {
                "needed_input": "pH, alkalinity, Mg, NH4, PO4 lab results",
                "current_status": "Missing from app inputs",
                "planned_metric": "Chemistry summary around dosing conditions and struvite risk windows",
                "why_it_matters": "Needed for mechanistic interpretation and bench/full-scale comparison.",
            },
            {
                "needed_input": "Bench-scale experiment result table",
                "current_status": "Missing",
                "planned_metric": "Dose-response comparison between bench and full-scale observations",
                "why_it_matters": "Needed to bridge dissertation Objectives 2-4.",
            },
        ]
    )


def parse_uploaded_csv(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file, parse_dates=True)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)


def normalize_optional_table(df):
    if df is None or df.empty:
        return df
    out = df.copy()
    for candidate in ["timestamp", "date", "sample_time", "sample_date"]:
        if candidate in out.columns:
            try:
                out[candidate] = pd.to_datetime(out[candidate], errors="coerce")
            except Exception:
                pass
    return out


def validate_optional_table_schema(df, title):
    schema = OPTIONAL_TABLE_SCHEMAS.get(title, {})
    if df is None or df.empty:
        return {
            "is_valid": False,
            "missing_required_groups": schema.get("required_any", []),
            "missing_recommended": schema.get("recommended", []),
        }

    columns = set(df.columns)
    missing_required_groups = [
        sorted(group) for group in schema.get("required_any", [])
        if columns.isdisjoint(group)
    ]
    missing_recommended = [
        col for col in schema.get("recommended", [])
        if col not in columns
    ]
    return {
        "is_valid": len(missing_required_groups) == 0,
        "missing_required_groups": missing_required_groups,
        "missing_recommended": missing_recommended,
    }


def render_optional_table_source(local_df, title, uploader_key, expected_columns):
    uploaded = st.file_uploader(
        f"Upload {title.lower()} CSV",
        type="csv",
        key=uploader_key,
        help=f"Optional CSV. Recommended columns include: {', '.join(expected_columns)}",
    )
    uploaded_df = normalize_optional_table(parse_uploaded_csv(uploaded))
    source_df = uploaded_df if uploaded_df is not None else normalize_optional_table(local_df)
    if source_df is None or source_df.empty:
        st.info(f"No {title.lower()} table found. Add `{title.lower().replace(' ', '_')}.csv` to `data/processed/` or upload one here.")
        return None

    validation = validate_optional_table_schema(source_df, title)
    if not validation["is_valid"]:
        missing_groups = [" / ".join(group) for group in validation["missing_required_groups"]]
        st.warning(
            f"{title} is missing required schema groups: {', '.join(missing_groups)}."
        )
    elif validation["missing_recommended"]:
        st.caption(
            f"{title} is usable, but recommended columns are missing: {', '.join(validation['missing_recommended'])}."
        )

    st.caption(f"Using {'uploaded' if uploaded_df is not None else 'local processed'} {title.lower()} table.")
    st.dataframe(source_df.head(200), use_container_width=True, height=220)
    return source_df


def render_struvite_placeholder(local_struvite_df, local_lab_df):
    render_section_intro(
        "Struvite And Lab Data Workflow",
        "This section now supports optional local or uploaded tables for scale observations and lab chemistry, while still making missing research inputs explicit when they are absent.",
    )
    render_help_tip("If you do not have structured struvite or lab tables yet, the tracker below stays visible so the gap remains explicit.")

    struvite_df = render_optional_table_source(
        local_struvite_df,
        "Struvite Observations",
        "struvite_upload",
        ["date", "location", "severity", "coupon_result", "notes"],
    )
    lab_df = render_optional_table_source(
        local_lab_df,
        "Chemistry Lab Results",
        "chem_lab_upload",
        ["date", "pH", "alkalinity", "mg", "nh4", "po4", "sample_location"],
    )

    if struvite_df is not None:
        st.markdown("**Struvite observation columns present**")
        st.write(", ".join(map(str, struvite_df.columns)))
    if lab_df is not None:
        st.markdown("**Chemistry lab columns present**")
        st.write(", ".join(map(str, lab_df.columns)))

    st.dataframe(build_struvite_placeholder_df(), use_container_width=True, height=220)


def render_variable_glossary():
    glossary_sections = {
        "Core odor signals": [
            ("`nh3_roll_mean_15min`", "15-minute rolling average of NH3. The dashboard uses this because NH3 is treated as a more sustained background signal, so the recent average is usually more informative than the single highest minute."),
            ("`h2s_roll_max_15min`", "15-minute rolling maximum of H2S. The dashboard uses this because H2S is treated as spike-driven, and short bursts can matter even when the average stays relatively low."),
            ("`nh3_nh3_ppm` / `h2s_h2s_ppm`", "Raw NH3 and H2S sensor readings in ppm."),
            ("`nh3_temperature_°f` / `h2s_temperature_°f`", "Sensor temperatures in degrees Fahrenheit."),
        ],
        "Rolling and lag features": [
            ("`roll_mean`", "Average over the last N minutes. Use it when you want the local typical level rather than point-to-point noise."),
            ("`roll_max`", "Maximum over the last N minutes. Use it when short peaks matter more than the average."),
            ("`lag_5min`, `lag_15min`, `lag_30min`, `lag_60min`", "The same signal shifted backward in time, used for time-history and predictive features."),
        ],
        "Flow and load variables": [
            ("`total_gpm`", "Total sludge/process flow in gallons per minute, summed across the main flow columns."),
            ("`lbs_per_min`", "Estimated volatile load transferred per minute, derived from flow using the project conversion factor."),
            ("`transferred_lbs_vol`", "Dashboard alias for `lbs_per_min`, mainly used in load-context charts."),
            ("`transferred_lbs_vol_daily`", "Daily total of minute-level transferred load."),
            ("`nh3_per_lb` / `h2s_per_lb`", "Odor signal normalized by load, used to compare intensity relative to throughput."),
        ],
        "Coverage and variability": [
            ("`n_obs_nh3`, `n_obs_h2s`, `n_obs_water`", "Number of non-missing observations available that day."),
            ("`nh3_coverage`, `h2s_coverage`, `water_coverage`", "Fraction of expected daily readings present. Higher coverage means more trustworthy summaries."),
            ("`nh3_std` / `h2s_std`", "Daily standard deviation, used as a simple measure of within-day variability."),
        ],
        "Chemistry and events": [
            ("`ferric_available` / `hcl_available`", "Binary flags indicating whether Ferric or HCl was active/available at that time."),
            ("`ferric_active_lbs_per_day`", "Estimated active ferric dose level per day when available in the data."),
            ("`Ferric_ON`, `Ferric_OFF`, `HCl_ON`, `HCl_OFF`", "Transition events detected when the binary chemistry flags switch on or off."),
        ],
        "Event-study metrics": [
            ("`event`", "A detected change point in plant chemistry status at a specific timestamp. In this app, events are Ferric ON, Ferric OFF, HCl ON, or HCl OFF."),
            ("`baseline`", "Typical pre-event level in the baseline window."),
            ("`post`", "Typical post-event level in the post window."),
            ("`delta`", "`post - baseline`. Negative means the signal fell after the event."),
            ("`percent_change`", "Percent change from baseline to post."),
            ("`time_to_min`", "Minutes after the event until the minimum post-event value is reached."),
            ("`persistence`", "How long the signal stayed below baseline after the event."),
            ("`post_iqr`", "Post-event spread, measured as the interquartile range."),
        ],
    }

    for section, items in glossary_sections.items():
        st.markdown(f"**{section}**")
        for term, explanation in items:
            st.markdown(f"- {term}: {explanation}")


def render_help_tip(text):
    if hasattr(st, "popover"):
        with st.popover("? Help"):
            st.write(text)
    else:
        st.caption(f"Help: {text}")


def render_section_intro(title, description):
    st.markdown(
        f"""
        <div class="section-intro">
            <div class="section-label">Section</div>
            <div class="section-title">{escape(title)}</div>
            <p class="section-copy">{escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
