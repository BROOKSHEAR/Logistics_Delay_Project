"""
Data loading module.

Unified data loading using paths from utils.paths.
"""
from __future__ import annotations

import pandas as pd

from logistics_delay.utils.paths import check_data_exists, DATA_PROCESSED
from logistics_delay.features.engineering import XGB_CAT_COLS


def load_raw_data() -> pd.DataFrame:
    """Load raw Excel data from data/raw/ without any processing.

    Returns:
        Raw DataFrame.
    """
    path = check_data_exists()
    df = pd.read_excel(path)
    print(f"[loader] Raw data loaded: {df.shape}")
    return df


def load_raw_data_with_target() -> pd.DataFrame:
    """Load raw data and build ``Answer`` target column.

    ``Answer`` = 0 (ontime == 'G') | 1 (otherwise, i.e., delayed).

    Returns:
        DataFrame with ``Answer`` column.
    """
    df = load_raw_data()
    df["Answer"] = df["ontime"].apply(lambda x: 0 if x == "G" else 1)
    counts = df["Answer"].value_counts()
    print(f"[loader] Target built: delayed={counts.get(1, 0)}, on-time={counts.get(0, 0)}")
    print(f"[loader] Delay rate: {counts.get(1, 0) / len(df) * 100:.2f}%")
    return df


def load_processed() -> pd.DataFrame:
    """Load preprocessed feature data (skip raw → engineer_features pipeline).

    Reads ``data/processed/truck_delay_handled_file.xlsx``,
    automatically sets ``category`` dtype for tree model categorical columns.

    Returns:
        DataFrame with ``Answer`` column and all features (6854×61).
    """
    path = DATA_PROCESSED / "truck_delay_handled_file.xlsx"
    df = pd.read_excel(path)

    # Excel does not preserve category dtype; restore manually
    for col in XGB_CAT_COLS:
        if col in df.columns:
            # Fill NaN + uniform str: prevent XGBoost errors from mixed types (int/str)
            # and CatBoost errors from NaN in categorical features
            df[col] = df[col].fillna("UNKNOWN").astype(str).astype("category")

    print(f"[loader] Preprocessed data loaded: {df.shape}")
    print(f"[loader] Delay rate: {df['Answer'].mean() * 100:.2f}%")
    return df
