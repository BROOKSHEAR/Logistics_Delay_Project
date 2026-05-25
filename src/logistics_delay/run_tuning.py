#!/usr/bin/env python3
"""
Run all hyperparameter tuning (two-stage + Optuna) and save results to CSV.

Usage:  python -m logistics_delay.run_tuning

Notebook 04_tuning.ipynb reads the saved CSVs for display — no need to
re-run tuning inside the notebook.
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
from logistics_delay.models.optuna_tuning import (
    run_optuna_search,
    save_optuna_results,
    suggest_lr_params,
    suggest_dt_params,
    suggest_rf_params,
    suggest_xgb_params,
    suggest_lgbm_params,
    suggest_cb_params,
)
from logistics_delay.models.train import temporal_split
from logistics_delay.features.engineering import FEATURES_XGB, XGB_CAT_COLS
from logistics_delay.utils.paths import SEED, TABLES_DIR

N_TRIALS = 100


def main():
    print("=" * 60)
    print("  Run all tuning — two-stage + Optuna")
    print("=" * 60)

    # ── 1. Data ──
    df = load_processed()
    X_train_enc, X_test_enc, y_train, y_test, _, _ = temporal_split(df, "enc")
    X_train_xgb, X_test_xgb, _, _, _, _ = temporal_split(df, "xgb")
    spw_t, spw_candidates = compute_and_print_spw(y_train)

    # ── 2. Two-stage: LR, DT, RF, XGB, LGBM ──
    print("\n" + "=" * 60)
    print("  Stage 1 — Two-Stage: LR, DT, RF, XGB, LGBM")
    print("=" * 60)

    lr_result = run_grid_search(
        LogisticRegression(max_iter=1000, random_state=SEED),
        lr_params_group, X_train_enc, y_train, X_test_enc, y_test,
    )
    dt_result = run_grid_search(
        DecisionTreeClassifier(random_state=SEED),
        dt_params, X_train_enc, y_train, X_test_enc, y_test,
    )
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

    # ── 3. Two-stage: CatBoost ──
    print("\n" + "=" * 60)
    print("  Stage 1b — Two-Stage: CatBoost")
    print("=" * 60)
    cat_feats = [c for c in XGB_CAT_COLS if c in FEATURES_XGB]
    cb_result = run_two_stage_search(
        CatBoostClassifier(random_seed=SEED, verbose=0, iterations=200),
        cb_params, CB_KEY_PARAMS, 20, X_train_xgb, y_train, X_test_xgb, y_test,
        fit_params={"cat_features": cat_feats},
    )

    # ── 4. Save two-stage ──
    stage2_df = pd.DataFrame(
        [lr_result, dt_result, rf_result, xgb_result, lgbm_result, cb_result]
    )
    save_tuning_results(stage2_df, TABLES_DIR)

    # ── 5. Optuna: LR, DT, RF, XGB, LGBM ──
    print("\n" + "=" * 60)
    print("  Stage 2 — Optuna: LR, DT, RF, XGB, LGBM")
    print("=" * 60)

    optuna_study_dir = TABLES_DIR / "optuna_studies"
    kw = {"random_state": SEED}
    optuna_lr_result = run_optuna_search(
        LogisticRegression, suggest_lr_params,
        X_train_enc, y_train, X_test_enc, y_test,
        n_trials=N_TRIALS, model_init_kwargs={**kw, "max_iter": 5000},
        save_study_path=str(optuna_study_dir / "LogisticRegression.pkl"),
    )
    optuna_dt_result = run_optuna_search(
        DecisionTreeClassifier, suggest_dt_params,
        X_train_enc, y_train, X_test_enc, y_test,
        n_trials=N_TRIALS, model_init_kwargs=kw,
        save_study_path=str(optuna_study_dir / "DecisionTreeClassifier.pkl"),
    )
    optuna_rf_result = run_optuna_search(
        RandomForestClassifier, suggest_rf_params,
        X_train_enc, y_train, X_test_enc, y_test,
        n_trials=N_TRIALS, model_init_kwargs={**kw, "n_jobs": -1},
        save_study_path=str(optuna_study_dir / "RandomForestClassifier.pkl"),
    )
    optuna_xgb_result = run_optuna_search(
        XGBClassifier, lambda t: suggest_xgb_params(t, spw_candidates=spw_candidates),
        X_train_xgb, y_train, X_test_xgb, y_test,
        n_trials=N_TRIALS, early_stopping_rounds=50,
        model_init_kwargs={**kw, "verbosity": 0, "enable_categorical": True, "n_jobs": -1},
        save_study_path=str(optuna_study_dir / "XGBClassifier.pkl"),
    )
    optuna_lgbm_result = run_optuna_search(
        LGBMClassifier, lambda t: suggest_lgbm_params(t, spw_candidates=spw_candidates),
        X_train_xgb, y_train, X_test_xgb, y_test,
        n_trials=N_TRIALS, early_stopping_rounds=50,
        model_init_kwargs={**kw, "verbose": -1, "n_jobs": -1},
        save_study_path=str(optuna_study_dir / "LGBMClassifier.pkl"),
    )

    # ── 6. Optuna: CatBoost ──
    print("\n" + "=" * 60)
    print("  Stage 2b — Optuna: CatBoost")
    print("=" * 60)
    optuna_cb_result = run_optuna_search(
        CatBoostClassifier, suggest_cb_params,
        X_train_xgb, y_train, X_test_xgb, y_test,
        n_trials=N_TRIALS, early_stopping_rounds=50,
        model_init_kwargs={"random_seed": SEED, "verbose": 0},
        fit_params={"cat_features": cat_feats},
        save_study_path=str(optuna_study_dir / "CatBoostClassifier.pkl"),
    )

    # ── 7. Save Optuna ──
    optuna_df = pd.DataFrame(
        [optuna_lr_result, optuna_dt_result, optuna_rf_result,
         optuna_xgb_result, optuna_lgbm_result, optuna_cb_result]
    )
    save_optuna_results(optuna_df, TABLES_DIR)

    # ── 8. Quick summary ──
    print("\n" + "=" * 60)
    print("  ALL DONE — results saved to:")
    print(f"    {TABLES_DIR / 'tuning_results.csv'}")
    print(f"    {TABLES_DIR / 'optuna_tuning_results.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
