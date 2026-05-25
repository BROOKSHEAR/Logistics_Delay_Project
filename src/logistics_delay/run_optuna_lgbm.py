#!/usr/bin/env python3
"""
LightGBM-only Optuna tuning.

Run this to generate the study pickle needed by notebook 04_tuning.ipynb
cell 11 for Optuna visualizations. Skips CatBoost (slow) and all other models.

Usage:  python -m logistics_delay.run_optuna_lgbm
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

warnings.filterwarnings("ignore")

import pandas as pd
from lightgbm import LGBMClassifier

from logistics_delay.data.loader import load_processed
from logistics_delay.models.optuna_tuning import (
    run_optuna_search,
    suggest_lgbm_params,
)
from logistics_delay.models.train import temporal_split
from logistics_delay.utils.paths import SEED, TABLES_DIR

N_TRIALS = 100


def main():
    print("=" * 60)
    print("  LightGBM-only Optuna tuning")
    print("=" * 60)

    # 1. Data
    df = load_processed()
    X_train_xgb, X_test_xgb, y_train, y_test, _, _ = temporal_split(df, "xgb")

    # 2. Compute scale_pos_weight candidates
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw_t = n_neg / n_pos if n_pos else 1.0
    spw_candidates = [spw_t * m for m in [0.5, 0.75, 1.0, 1.25, 1.5]]

    # 3. Run Optuna for LGBM only (100 trials)
    optuna_study_dir = TABLES_DIR / "optuna_studies"
    optuna_study_dir.mkdir(parents=True, exist_ok=True)

    optuna_lgbm_result = run_optuna_search(
        LGBMClassifier, lambda t: suggest_lgbm_params(t, spw_candidates=spw_candidates),
        X_train_xgb, y_train, X_test_xgb, y_test,
        n_trials=N_TRIALS, early_stopping_rounds=50,
        model_init_kwargs={"random_state": SEED, "verbose": -1, "n_jobs": -1},
        save_study_path=str(optuna_study_dir / "LGBMClassifier.pkl"),
    )

    # 4. Update CSV: keep existing rows, update/replace LGBM row
    csv_path = TABLES_DIR / "optuna_tuning_results.csv"
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        print(f"\nExisting CSV: {len(existing)} models loaded")
        # Remove old LGBM row if present
        existing = existing[existing["model"] != "LGBMClassifier"]
    else:
        existing = pd.DataFrame()

    new_row = pd.DataFrame([optuna_lgbm_result])
    updated = pd.concat([existing, new_row], ignore_index=True)

    # Format best_params column for CSV readability
    if "best_params" in updated.columns:
        updated["best_params"] = updated["best_params"].apply(
            lambda d: ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in d.items()
            ) if isinstance(d, dict) else str(d)
        )

    updated.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] CSV updated -> {csv_path}")
    print(f"    LGBM CV AUC  = {optuna_lgbm_result['cv_auc'] * 100:.2f}%")
    print(f"    LGBM Test AUC = {optuna_lgbm_result['test_auc'] * 100:.2f}%")

    print("\n" + "=" * 60)
    print("  Done! Now re-run cell 11 in 04_tuning.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()
