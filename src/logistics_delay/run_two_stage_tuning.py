#!/usr/bin/env python3
"""
Two-stage hyperparameter tuning — all 6 models.

Stage 1: LogisticRegression + DecisionTree via GridSearchCV (exhaustive).
Stage 2: RandomForest, XGBoost, LightGBM, CatBoost via RandomizedSearchCV
         coarse + fine GridSearchCV on top-3 key params.

Results saved to outputs/tables/tuning_results.csv

Usage:  python -m logistics_delay.run_two_stage_tuning
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from logistics_delay.data.loader import load_processed
from logistics_delay.models.tuning import (
    compute_and_print_spw,
    run_grid_search,
    run_two_stage_search,
    save_tuning_results,
    lr_params_group,
    dt_params,
    rf_params,
    xgb_params,
    lgbm_params,
    cb_params,
    RF_KEY_PARAMS,
    XGB_KEY_PARAMS,
    LGBM_KEY_PARAMS,
    CB_KEY_PARAMS,
)
from logistics_delay.models.train import temporal_split
from logistics_delay.features.engineering import FEATURES_XGB, XGB_CAT_COLS
from logistics_delay.utils.paths import SEED, TABLES_DIR


def main():
    print("=" * 60)
    print("  Two-Stage Tuning — All 6 Models")
    print("=" * 60)

    # ── 1. Data ──
    df = load_processed()
    X_train_enc, X_test_enc, y_train, y_test, _, _ = temporal_split(df, "enc")
    X_train_xgb, X_test_xgb, _, _, _, _ = temporal_split(df, "xgb")
    spw_t, spw_candidates = compute_and_print_spw(y_train)

    # ── 2. LR + DT (exhaustive GridSearch) ──
    print("\n" + "=" * 60)
    print("  Stage 1 — GridSearch: LR, DT")
    print("=" * 60)

    lr_result = run_grid_search(
        LogisticRegression(max_iter=1000, random_state=SEED),
        lr_params_group, X_train_enc, y_train, X_test_enc, y_test,
    )
    dt_result = run_grid_search(
        DecisionTreeClassifier(random_state=SEED),
        dt_params, X_train_enc, y_train, X_test_enc, y_test,
    )

    # ── 3. RF, XGB, LGBM (two-stage) ──
    print("\n" + "=" * 60)
    print("  Stage 2 — Two-Stage: RF, XGB, LGBM")
    print("=" * 60)

    rf_result = run_two_stage_search(
        RandomForestClassifier(random_state=SEED), rf_params,
        RF_KEY_PARAMS, 30, X_train_enc, y_train, X_test_enc, y_test,
    )
    xgb_params_local = {**xgb_params, "scale_pos_weight": spw_candidates}
    xgb_result = run_two_stage_search(
        XGBClassifier(random_state=SEED, verbosity=0, enable_categorical=True),
        xgb_params_local, XGB_KEY_PARAMS, 40, X_train_xgb, y_train, X_test_xgb, y_test,
    )
    lgbm_result = run_two_stage_search(
        LGBMClassifier(random_state=SEED, verbose=-1),
        lgbm_params, LGBM_KEY_PARAMS, 40, X_train_xgb, y_train, X_test_xgb, y_test,
    )

    # ── 4. CatBoost (two-stage, with categorical features) ──
    print("\n" + "=" * 60)
    print("  Stage 3 — Two-Stage: CatBoost")
    print("=" * 60)
    cat_feats = [c for c in XGB_CAT_COLS if c in FEATURES_XGB]
    cb_result = run_two_stage_search(
        CatBoostClassifier(random_seed=SEED, verbose=0, iterations=200),
        cb_params, CB_KEY_PARAMS, 20, X_train_xgb, y_train, X_test_xgb, y_test,
        fit_params={"cat_features": cat_feats},
    )

    # ── 5. Save ──
    stage2_df = pd.DataFrame(
        [lr_result, dt_result, rf_result, xgb_result, lgbm_result, cb_result]
    )
    save_tuning_results(stage2_df, TABLES_DIR)

    print("\n" + "=" * 60)
    print("  ALL DONE — results saved to:")
    print(f"    {TABLES_DIR / 'tuning_results.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
