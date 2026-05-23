"""
Model training module.

Provides data splitting functions and single/multi-model training wrappers.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from logistics_delay.utils.paths import RANDOM_STATE
from logistics_delay.features.engineering import (
    FEATURES_ENC, FEATURES_XGB, get_feature_lists,
)


# ──────────────── Data splitting functions ────────────────

def random_split(df: pd.DataFrame,
                 feature_set: str = "enc",
                 test_size: float = 0.2,
                 stratify: bool = True):
    """Random split (80/20 stratified).

    Args:
        df: Full DataFrame with ``Answer`` column and feature columns.
        feature_set: ``"enc"`` or ``"xgb"``.
        test_size: Test set ratio.
        stratify: Whether to stratify by target variable.

    Returns:
        (X_train, X_test, y_train, y_test, scale_pos_weight) or
        (X_train, X_test, y_train, y_test, scale_pos_weight, X_train_xgb, X_test_xgb)
        if feature_set="both".
    """
    if feature_set == "xgb":
        features = FEATURES_XGB
    else:
        features = FEATURES_ENC

    X = df[features]
    y = df["Answer"]

    stratify_y = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify_y,
    )
    spw = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    print(f"[train] Random split: train {len(y_train)} / test {len(y_test)}")
    print(f"[train] scale_pos_weight = {spw:.4f}")

    return X_train, X_test, y_train, y_test, spw


def temporal_split(df: pd.DataFrame,
                   feature_set: str = "enc",
                   test_size: float = 0.2):
    """Temporal split: sort by ``trip_start_date``, first ``(1-test_size)`` train, last ``test_size`` test.

    Ensures all training data precedes test data to avoid leakage.

    Args:
        df: DataFrame with ``Answer``, ``trip_start_date`` and features.
        feature_set: ``"enc"`` or ``"xgb"``.

    Returns:
        (X_train, X_test, y_train, y_test, scale_pos_weight, split_date)
    """
    if feature_set == "xgb":
        features = FEATURES_XGB
    else:
        features = FEATURES_ENC

    df_sorted = df.sort_values("trip_start_date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    cutoff_date = df_sorted.iloc[split_idx]["trip_start_date"]

    X_train = df_sorted.loc[: split_idx - 1, features].reset_index(drop=True)
    X_test = df_sorted.loc[split_idx:, features].reset_index(drop=True)
    y_train = df_sorted.loc[: split_idx - 1, "Answer"].reset_index(drop=True)
    y_test = df_sorted.loc[split_idx:, "Answer"].reset_index(drop=True)
    spw = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    print(f"[train] Temporal split: cutoff {cutoff_date.date()}")
    print(f"[train] Train: {len(y_train)} records, "
          f"{df_sorted.loc[:split_idx - 1, 'trip_start_date'].min().date()} ~ "
          f"{df_sorted.loc[:split_idx - 1, 'trip_start_date'].max().date()}")
    print(f"[train] Test: {len(y_test)} records, "
          f"{df_sorted.loc[split_idx, 'trip_start_date'].date()} ~ "
          f"{df_sorted['trip_start_date'].max().date()}")
    print(f"[train] scale_pos_weight = {spw:.4f}")

    return X_train, X_test, y_train, y_test, spw, cutoff_date


# ──────────────── Training wrapper ────────────────

def train_model(model, X_train, y_train, **fit_kwargs):
    """Train a single model.

    Args:
        model: sklearn-compatible model instance.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Fitted model.
    """
    model.fit(X_train, y_train, **fit_kwargs)
    return model
