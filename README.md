# Wastewater Odor Analytics Platform

## Overview

This project is an end-to-end data pipeline and analytics dashboard for monitoring and analyzing odor emissions (NH₃ and H₂S) in wastewater treatment operations.

It integrates:
- Multi-source sensor data (gas + operational)
- Time-series preprocessing and feature engineering
- Event detection for chemical dosing changes
- Event-based statistical analysis (pre/post effects)
- Interactive visualization via Streamlit

---

## Repository Structure

    .
    ├── app/
    │   ├── app.py
    │   ├── plots.py
    │   └── __init__.py
    │
    ├── scripts/
    │   ├── load_data.py
    │   ├── preprocess.py
    │   ├── features.py
    │   ├── chemistry_features.py
    │   ├── events.py
    │   ├── build_master.py
    │   ├── build_daily.py
    │   ├── build_aggregates.py
    │   ├── analytics.py
    │   ├── event_metrics.py
    │   ├── event_study.py
    │   ├── event_window_timeseries.py
    │   ├── full_timeseries_plots.py
    │   ├── multi_panel_comparison.py
    │   ├── explore.py
    │   ├── paths.py
    │   └── __init__.py
    │
    ├── data/
    │   ├── raw/
    │   └── processed/
    │
    ├── figures/
    ├── plots/
    ├── archive/
    ├── notes/
    │
    ├── config.py
    ├── requirements.txt
    ├── .gitignore
    └── README.md

---

## Installation

### 1. Clone the repository

git clone <your-repo-url>  
cd wastewater-odor-analytics  

---

### 2. Create virtual environment

macOS / Linux:

python -m venv venv  
source venv/bin/activate  

Windows:

python -m venv venv  
venv\Scripts\activate  

---

### 3. Install dependencies

pip install -r requirements.txt  

---

## Data Setup

Create required directories:

mkdir -p data/raw data/processed  

Place raw sensor files in:

data/raw/

Expected inputs:
- NH₃ sensor CSV exports
- H₂S sensor CSV exports
- Operational flow data

---

## Pipeline Execution

Run the pipeline:

python -m scripts.build_master  
python -m scripts.build_daily  
python -m scripts.build_aggregates  

Optional analysis:

python -m scripts.event_metrics  
python -m scripts.event_study  
python -m scripts.full_timeseries_plots  
python -m scripts.multi_panel_comparison  

---

## Running the Dashboard

streamlit run app/app.py  

---

## Key Concepts

### Signals
- NH₃: continuous → rolling mean
- H₂S: spike-driven → rolling max

### Event Detection
- Ferric chloride dosing
- Hydrochloric acid (HCl) dosing

### Event Windows
- Baseline: −48h to −12h  
- Post: +12h to +96h  

### Metrics
- Median change
- Percent change
- Time to minimum
- Persistence below baseline
- Variability (IQR)

---

## Outputs

- master_1min.parquet  
- master_daily.parquet  
- monthly_summary.parquet  
- weekday_summary.parquet  
- event metrics CSV  
- Plotly visualizations  

---

## Notes

- Data is excluded from version control
- Assumes 1-minute time resolution
- Missing data handled via interpolation

---

## Author

Mutsa Mungoshi  
Applied Data Science | Time-Series Analytics | Wastewater Systems