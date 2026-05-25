#!/usr/bin/env python3
"""
Run ablation, model comparison, and CatBoost SHAP analysis, saving all results
to CSV / NPY / PKL files. Notebooks 05 / 06 load these files for display.

Usage:  python -m logistics_delay.run_analysis
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from logistics_delay.ablation.ablation import (
    load_and_prepare_data,
    run_feature_ablation,
    run_learning_curves,
    run_geo_ablation,
    save_results,
    save_geo_results,
)
from logistics_delay.features.engineering import FEATURES_XGB, XGB_CAT_COLS
from logistics_delay.models.comparison import run_comparison, _create_model, _select_features
from logistics_delay.interpretation.shap_analysis import compute_shap_importance
from logistics_delay.utils.paths import TABLES_DIR, FIGURES_SHAP, ensure_output_dirs

ensure_output_dirs()

# ── Best CatBoost params (from tuning) ──
BEST_CB_PARAMS = {
    "learning_rate": 0.05,
    "depth": 8,
    "iterations": 100,
    "l2_leaf_reg": 5,
    "border_count": 64,
    "bagging_temperature": 1,
    "random_strength": 0,
}


def main():
    # ════════════════════════════════════════════════════════════════
    #  1. Ablation experiments
    # ════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  Ablation — best params")
    print("=" * 60)

    df_sorted = load_and_prepare_data()

    ablation_results = run_feature_ablation(
        df_sorted,
        feature_list=FEATURES_XGB,
        catboost_params=BEST_CB_PARAMS,
    )
    lc_results = run_learning_curves(
        df_sorted,
        feature_list=FEATURES_XGB,
        ablation_df=ablation_results,
        catboost_params=BEST_CB_PARAMS,
    )
    save_results(ablation_results, lc_results, save_dir=TABLES_DIR)

    geo_results = run_geo_ablation(df_sorted, catboost_params=BEST_CB_PARAMS)
    save_geo_results(geo_results, save_dir=TABLES_DIR)

    geo_results_default = run_geo_ablation(df_sorted, catboost_params={"iterations": 200})
    geo_results_default.to_csv(
        TABLES_DIR / "geo_ablation_default_results.csv",
        index=False, encoding="utf-8-sig",
    )
    print(f"[OK] Default geo ablation → {TABLES_DIR / 'geo_ablation_default_results.csv'}")

    # ════════════════════════════════════════════════════════════════
    #  2. Model comparison — TimeSeriesSplit 5-fold CV
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Model comparison — TimeSeriesSplit 5-fold")
    print("=" * 60)

    # Load data again (separate entry point to keep modules independent)
    from logistics_delay.data.loader import load_processed
    df = load_processed()

    results = run_comparison(df, n_bootstrap=2000)
    results["auc_ci"].to_csv(
        TABLES_DIR / "auc_ci_comparison.csv", index=False, encoding="utf-8-sig",
    )
    results["rankings_df"].to_csv(
        TABLES_DIR / "model_rankings.csv", encoding="utf-8-sig",
    )
    results["win_matrix"].to_csv(
        TABLES_DIR / "win_matrix.csv", encoding="utf-8-sig",
    )
    # fold_aucs needed by notebook scatter plot
    results["fold_aucs"].to_csv(
        TABLES_DIR / "fold_aucs.csv", encoding="utf-8-sig",
    )

    # ════════════════════════════════════════════════════════════════
    #  3. CatBoost SHAP (temporal split)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  CatBoost SHAP — temporal split")
    print("=" * 60)

    SHAP_DROP_FEATURES = ["planned_days_enc", "GpsProvider"]

    df_sorted = df.sort_values("trip_start_date").reset_index(drop=True)
    si = int(len(df_sorted) * 0.8)
    X_train_xgb = _select_features("XGBoost", df_sorted.loc[:si - 1]).drop(columns=SHAP_DROP_FEATURES, errors="ignore")
    X_test_xgb = _select_features("XGBoost", df_sorted.loc[si:]).drop(columns=SHAP_DROP_FEATURES, errors="ignore")
    y_train_t = df_sorted.loc[:si - 1, "Answer"].reset_index(drop=True)
    y_test_t = df_sorted.loc[si:, "Answer"].reset_index(drop=True)
    spw_t = (y_train_t == 0).sum() / (y_train_t == 1).sum()
    test_ids = df_sorted.loc[si:, "BookingID"].reset_index(drop=True)
    shap_cat_cols = [c for c in XGB_CAT_COLS if c not in SHAP_DROP_FEATURES]
    print(f"Temporal split: cutoff {df_sorted.iloc[si]['trip_start_date'].date()}")
    print(f"  Train: {si}  Test: {len(df_sorted) - si}  spw={spw_t:.4f}")
    print(f"  Dropped leakage features: {SHAP_DROP_FEATURES}")

    cb = _create_model("CatBoost", spw_t)
    cb.fit(X_train_xgb, y_train_t, cat_features=shap_cat_cols)
    cb_auc = roc_auc_score(y_test_t, cb.predict_proba(X_test_xgb)[:, 1])
    print(f"CatBoost AUC (temporal): {cb_auc * 100:.2f}%")

    imp_cb, shap_v_cb, exp_cb = compute_shap_importance(cb, X_test_xgb, model_type="tree")

    print(f"\n[OK] SHAP computed (not saved) — notebook trains inline")

    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
