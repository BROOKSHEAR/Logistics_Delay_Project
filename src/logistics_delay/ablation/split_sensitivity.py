"""
Split point sensitivity analysis: test whether CatBoost test_auc is stable under single temporal split.

Iterate over multiple training set ratios (70%, 75%, 80%, 85%),
train CatBoost once per ratio (using best hyperparams),
observe test_auc sensitivity to split point.

Usage:
    python -m src.logistics_delay.ablation.split_sensitivity

Outputs:
    - outputs/tables/split_sensitivity.csv
    - console comparison table
"""
from __future__ import annotations

import tempfile
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from catboost import CatBoostClassifier

from logistics_delay.ablation.ablation import load_and_prepare_data
from logistics_delay.features.engineering import FEATURES_XGB, XGB_CAT_COLS
from logistics_delay.utils.paths import SEED, TABLES_DIR

warnings.filterwarnings("ignore")

# Best CatBoost params (from 04_tuning.ipynb)
BEST_CB_PARAMS: dict = {
    "learning_rate": 0.05,
    "depth": 8,
    "iterations": 100,
    "l2_leaf_reg": 5,
    "border_count": 64,
    "bagging_temperature": 1,
    "random_strength": 0,
}


def run_sensitivity(
    df_sorted: pd.DataFrame,
    split_ratios: list[float] | None = None,
) -> pd.DataFrame:
    """Train CatBoost for each split ratio and record test_auc.

    Args:
        df_sorted: Full DataFrame sorted by trip_start_date.
        split_ratios: Training ratios to test, default [0.70, 0.75, 0.80, 0.85].

    Returns:
        DataFrame, one row per split ratio, with:
        - train_ratio: Training ratio
        - test_ratio: Test ratio
        - train_range: Training date range
        - test_range: Test date range
        - train_size / test_size
        - pos_rate_train / pos_rate_test: Delay rates
        - spw: scale_pos_weight
        - test_auc / test_f1
    """
    if split_ratios is None:
        split_ratios = [0.70, 0.75, 0.80, 0.85]

    feature_list = FEATURES_XGB
    cat_feats = [c for c in XGB_CAT_COLS if c in feature_list]

    records: list[dict] = []

    for ratio in split_ratios:
        split_idx = int(len(df_sorted) * ratio)

        X_train = df_sorted.loc[: split_idx - 1, feature_list].reset_index(drop=True)
        X_test = df_sorted.loc[split_idx:, feature_list].reset_index(drop=True)
        y_train = df_sorted.loc[: split_idx - 1, "Answer"].reset_index(drop=True)
        y_test = df_sorted.loc[split_idx:, "Answer"].reset_index(drop=True)

        spw = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)

        start_date = df_sorted["trip_start_date"]
        train_range = (
            f"{start_date.iloc[0].date()} ~ {start_date.iloc[split_idx - 1].date()}"
        )
        test_range = (
            f"{start_date.iloc[split_idx].date()} ~ {start_date.iloc[-1].date()}"
        )

        model = CatBoostClassifier(
            **BEST_CB_PARAMS,
            class_weights={0: 1.0, 1: spw},
            random_seed=SEED,
            verbose=0,
            train_dir=tempfile.gettempdir(),
            cat_features=cat_feats,
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        records.append({
            "train_ratio": ratio,
            "test_ratio": round(1 - ratio, 2),
            "train_range": train_range,
            "test_range": test_range,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "pos_rate_train": round(float(y_train.mean()), 4),
            "pos_rate_test": round(float(y_test.mean()), 4),
            "spw": round(spw, 2),
            "test_auc": round(roc_auc_score(y_test, y_prob), 4),
            "test_f1": round(f1_score(y_test, y_pred), 4),
        })

    return pd.DataFrame(records)


def print_results(df: pd.DataFrame) -> None:
    """Pretty-print sensitivity analysis results."""
    print("=" * 64)
    print("  CatBoost split point sensitivity analysis")
    print("=" * 64)
    print(
        f"  {'Train':>6s}  {'Test':>6s}  {'Train delay':<10s}"
        f"  {'Test delay':<10s}  {'SPW':>6s}  {'AUC':>7s}  {'F1':>7s}"
    )
    print("  " + "-" * 58)
    for _, row in df.iterrows():
        print(
            f"  {row['train_ratio']:>5.0%}  {row['test_ratio']:>5.0%}  "
            f"{row['pos_rate_train']:>7.2%}   {row['pos_rate_test']:>7.2%}   "
            f"{row['spw']:>6.2f}  {row['test_auc']:>7.4f}  {row['test_f1']:>7.4f}"
        )
    print("  " + "-" * 58)

    aucs = df["test_auc"].values
    print(f"  AUC mean: {aucs.mean():.4f}  "
          f"std: {aucs.std():.4f}  "
          f"range: {aucs.max() - aucs.min():.4f}")
    print(f"  Interpretation: ", end="")
    if aucs.max() - aucs.min() < 0.01:
        print("[OK] AUC variation < 1pp, split point has little effect, single split OK.")
    elif aucs.max() - aucs.min() < 0.02:
        print("[WARN] AUC variation 1~2pp, moderate sensitivity, consider TimeSeriesSplit.")
    else:
        print("[ALERT] AUC variation > 2pp, highly sensitive, must use TimeSeriesSplit.")

    print("\n  Per-split date ranges:")
    for _, row in df.iterrows():
        print(f"    {row['train_ratio']:.0%}/{-row['test_ratio']:.0%}  "
              f"Train: {row['train_range']}")
        print(f"    {'':>6s}  Test: {row['test_range']}")


def save_results(df: pd.DataFrame) -> None:
    """Save to CSV."""
    import os
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = TABLES_DIR / "split_sensitivity.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Sensitivity analysis results → {path}")


if __name__ == "__main__":
    print("Loading data...")
    df_sorted = load_and_prepare_data()

    result_df = run_sensitivity(df_sorted)
    print_results(result_df)
    save_results(result_df)
