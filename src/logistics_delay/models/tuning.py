"""
Two-stage hyperparameter tuning module.

Stage 1: LogisticRegression + DecisionTree with GridSearchCV + TimeSeriesSplit exhaustive search.
Stage 2: RandomForest + XGBoost + LightGBM + CatBoost with RandomizedSearchCV coarse
          + fine GridSearchCV (3 key params × 3 candidates = ≤27 combos).

Usage:
    from logistics_delay.models.tuning import (
        compute_and_print_spw, run_grid_search, run_two_stage_search, refine_grid,
        lr_params_group, dt_params, rf_params, xgb_params, lgbm_params, cb_params,
        RF_KEY_PARAMS, XGB_KEY_PARAMS, LGBM_KEY_PARAMS, CB_KEY_PARAMS,
    )
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from logistics_delay.utils.paths import SEED, TABLES_DIR
from logistics_delay.features.engineering import FEATURES_XGB, XGB_CAT_COLS

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
#  Parameter space definition
# ════════════════════════════════════════════════════════════════

# ── LogisticRegression: three param_grids to avoid illegal penalty/solver combos ──

_LR_C = [0.001, 0.01, 0.1, 1, 10, 100]
_LR_CW = [None, "balanced"]
_LR_MI = [1000, 2000]

lr_params_group = [
    {
        "penalty": ["l1"],
        "solver": ["saga"],
        "C": _LR_C,
        "class_weight": _LR_CW,
        "max_iter": _LR_MI,
    },
    {
        "penalty": ["l2"],
        "solver": ["lbfgs", "saga"],
        "C": _LR_C,
        "class_weight": _LR_CW,
        "max_iter": _LR_MI,
    },
    {
        "penalty": ["elasticnet"],
        "solver": ["saga"],
        "C": _LR_C,
        "class_weight": _LR_CW,
        "max_iter": _LR_MI,
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
]

# ── DecisionTree ──

dt_params = {
    "max_depth": [3, 4, 5, 6, 8],
    "min_samples_split": [10, 20, 40],
    "min_samples_leaf": [5, 10, 20, 50],
    "criterion": ["gini", "entropy"],
    "class_weight": ["balanced", None],
}

# ── RandomForest (full param space) ──

rf_params = {
    "n_estimators": [50, 100, 200, 300, 500],
    "max_depth": [3, 5, 7, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
    "class_weight": ["balanced", "balanced_subsample", None],
}

# ── XGBoost (full param space, scale_pos_weight injected dynamically by spw_t) ──

xgb_params = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "n_estimators": [100, 200, 300, 500],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "gamma": [0, 0.1, 0.5, 1],
    "reg_alpha": [0, 0.1, 1, 10],
    "reg_lambda": [0.1, 1, 10, 100],
}

# ── LightGBM (full param space) ──

lgbm_params = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "num_leaves": [15, 31, 63, 127, 255],
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [-1, 5, 10, 15, 20],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_samples": [5, 10, 20, 50],
    "reg_alpha": [0, 0.1, 1, 10],
    "reg_lambda": [0.1, 1, 10, 100],
}

# ── CatBoost (full param space) ──

cb_params = {
    "learning_rate": [0.03, 0.05, 0.1],
    "depth": [4, 6, 8],
    "iterations": [100, 200],
    "l2_leaf_reg": [1, 3, 5],
    "border_count": [64, 128],
    "bagging_temperature": [0, 0.5, 1],
    "random_strength": [0, 0.5, 1],
}
# All models have max_depth/depth capped at 8; verify with learning curves
# ── Key params for Stage 2 fine search (top 3 per model) ──

RF_KEY_PARAMS = ["n_estimators", "max_depth", "min_samples_leaf"]
XGB_KEY_PARAMS = ["learning_rate", "max_depth", "n_estimators"]
LGBM_KEY_PARAMS = ["learning_rate", "num_leaves", "n_estimators"]
CB_KEY_PARAMS = ["learning_rate", "depth", "iterations"]


# ════════════════════════════════════════════════════════════════
#  Utility functions
# ════════════════════════════════════════════════════════════════

def compute_and_print_spw(y_train, n_splits=5):
    """Compute spw and print training set sample distribution and per-fold delay rate.

    Args:
        y_train: Training labels (pd.Series or array-like).
        n_splits: Number of TimeSeriesSplit folds.

    Returns:
        (spw_t, spw_candidates) tuple:
        - spw_t: negative count / positive count
        - spw_candidates: [spw_t*0.5, 0.75, 1.0, 1.25, 1.5]
    """
    print("=" * 60)
    print("  Sample distribution and scale_pos_weight computation")
    print("=" * 60)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    total = len(y_train)
    spw_t = n_neg / n_pos if n_pos > 0 else 1.0

    print(f"\nTraining set sample distribution:")
    print(f"  Total: {total}")
    print(f"  Positive (delayed): {n_pos} ({n_pos / total * 100:.2f}%)")
    print(f"  Negative (on-time): {n_neg} ({n_neg / total * 100:.2f}%)")
    print(f"  spw_t (neg/pos) = {spw_t:.4f}")

    print(f"\nTimeSeriesSplit({n_splits} folds) per-fold delay rate:")
    print(f"  {'Fold':<6} {'Train delay':<14} {'Val delay':<14} {'Val pos rate'}")
    print(f"  {'-' * 54}")

    dummy_X = np.zeros(len(y_train))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for i, (tr_idx, val_idx) in enumerate(tscv.split(dummy_X, y_train)):
        y_tr = y_train.iloc[tr_idx] if hasattr(y_train, "iloc") else y_train[tr_idx]
        y_vl = y_train.iloc[val_idx] if hasattr(y_train, "iloc") else y_train[val_idx]
        tr_rate = float(y_tr.mean()) * 100
        vl_rate = float(y_vl.mean()) * 100
        vl_pos = float(y_vl.sum() / max(len(y_vl), 1)) * 100
        print(f"  Fold {i:<2}  {tr_rate:<12.2f}%  {vl_rate:<12.2f}%  {vl_pos:.2f}%")

    spw_candidates = [spw_t * m for m in [0.5, 0.75, 1.0, 1.25, 1.5]]
    print(f"\nDynamically generated scale_pos_weight candidates:")
    print(f"  {', '.join(f'{v:.4f}' for v in spw_candidates)}")
    print(f"  Rule: [spw_t×0.5, spw_t×0.75, spw_t×1.0, spw_t×1.25, spw_t×1.5]")

    return spw_t, spw_candidates


def refine_grid(best_params, param_dist, key_params):
    """Generate fine grid from coarse search best_params.

    For each param in ``key_params``, take the best value and its left/right
    neighbors from ``param_dist`` (≤3 values total); at boundaries, expand
    toward available values. Non-key params are fixed to ``best_params``.

    Args:
        best_params: ``best_params_`` from coarse search (dict).
        param_dist: Coarse search param distribution (dict, values must be list-like).
        key_params: Names of the 3 key params for fine search.

    Returns:
        dict, usable as ``param_grid`` for ``GridSearchCV``.
    """
    fine_grid = {}

    for param in key_params:
        candidates = list(param_dist.get(param, []))
        if not candidates:
            fine_grid[param] = [best_params[param]]
            continue

        best_val = best_params[param]
        # Find position of best value in candidate list
        try:
            idx = candidates.index(best_val)
        except ValueError:
            idx = min(
                range(len(candidates)),
                key=lambda i: abs(candidates[i] - best_val),
            )

        n = len(candidates)
        start = max(0, min(idx - 1, n - 3))
        end = min(n, start + 3)
        if end - start < 3:
            start = max(0, end - 3)

        fine_grid[param] = candidates[start:end]

    # Non-key params fixed to best value
    for param, val in best_params.items():
        if param not in key_params:
            fine_grid[param] = [val]

    return fine_grid


# ════════════════════════════════════════════════════════════════
#  Stage 1: GridSearchCV exhaustive
# ════════════════════════════════════════════════════════════════

def run_grid_search(model, param_grid, X_train, y_train, X_test, y_test):
    """Exhaustive search with ``GridSearchCV`` + ``TimeSeriesSplit(n_splits=5)``.

    Args:
        model: Unfitted sklearn model instance.
        param_grid: ``param_grid`` for ``GridSearchCV`` (dict or list of dicts).
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        dict with keys ``model``, ``cv_auc``, ``test_auc``, ``test_f1``, ``best_params``.
    """
    model_name = type(model).__name__
    tscv = TimeSeriesSplit(n_splits=5)

    print(f"\n{'=' * 60}")
    print(f"  GridSearch: {model_name}")
    print(f"{'=' * 60}")

    gs = GridSearchCV(model, param_grid, cv=tscv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    result = {
        "model": model_name,
        "cv_auc": gs.best_score_,
        "test_auc": roc_auc_score(y_test, y_prob),
        "test_f1": f1_score(y_test, y_pred),
        "best_params": gs.best_params_,
    }

    print(f"  CV AUC  = {result['cv_auc'] * 100:.2f}%")
    print(f"  Test AUC = {result['test_auc'] * 100:.2f}%")
    print(f"  Test F1  = {result['test_f1']:.4f}")

    return result


# ════════════════════════════════════════════════════════════════
#  Stage 2: RandomizedSearchCV coarse → GridSearchCV fine
# ════════════════════════════════════════════════════════════════

def run_two_stage_search(
    model,
    param_dist,
    key_params,
    n_iter,
    X_train,
    y_train,
    X_test,
    y_test,
    fit_params=None,
    seed=SEED,
):
    """Two-stage search: ``RandomizedSearchCV`` coarse + fine ``GridSearchCV``.

    Fine search only explores neighbor values around the best for ``key_params``;
    all other params are fixed to ``best_params_`` from coarse search (≤27 combos).

    Args:
        model: Unfitted model instance.
        param_dist: Coarse search param distribution (dict, values must be list-like).
        key_params: List of 3 key param names for fine search.
        n_iter: ``RandomizedSearchCV`` sampling iterations.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        fit_params: Extra params for ``.fit()`` (e.g., CatBoost ``cat_features``).
        seed: Random seed.

    Returns:
        dict with keys ``model``, ``cv_auc``, ``test_auc``, ``test_f1``, ``best_params``.
    """
    model_name = type(model).__name__
    tscv = TimeSeriesSplit(n_splits=5)

    # ── Stage 1: Coarse search ──
    print(f"\n{'=' * 60}")
    print(f"  Stage 1/2 — Random search: {model_name} (n_iter={n_iter})")
    print(f"{'=' * 60}")

    rs = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-2,
        random_state=seed,
    )
    rs.fit(X_train, y_train, **(fit_params or {}))
    print(f"  [Coarse] Best CV AUC = {rs.best_score_ * 100:.2f}%")
    for k, v in rs.best_params_.items():
        print(f"    {k}: {v}")

    # ── Stage 2: Fine search ──
    print(f"\n{'=' * 60}")
    print(f"  Stage 2/2 — Fine grid search: {model_name}")
    print(f"{'=' * 60}")

    fine_grid = refine_grid(rs.best_params_, param_dist, key_params)
    total_combos = 1
    for k in key_params:
        total_combos *= len(fine_grid[k])
    print(f"  Fine grid: {len(key_params)} key params × {total_combos} combos:")
    for k in key_params:
        print(f"    {k}: {fine_grid[k]}")
    for k, v in fine_grid.items():
        if k not in key_params:
            print(f"    {k}: fixed to {v[0]}")

    stage2_model = clone(rs.best_estimator_)
    gs = GridSearchCV(
        stage2_model,
        fine_grid,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    gs.fit(X_train, y_train, **(fit_params or {}))
    print(f"  [Fine] Best CV AUC = {gs.best_score_ * 100:.2f}%")

    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    result = {
        "model": model_name,
        "cv_auc": gs.best_score_,
        "test_auc": roc_auc_score(y_test, y_prob),
        "test_f1": f1_score(y_test, y_pred),
        "best_params": gs.best_params_,
    }

    print(f"\n  [OK] {model_name} complete")
    print(f"     CV AUC  = {result['cv_auc'] * 100:.2f}%")
    print(f"     Test AUC = {result['test_auc'] * 100:.2f}%")
    print(f"     Test F1  = {result['test_f1']:.4f}")

    return result


# ════════════════════════════════════════════════════════════════
#  Save results
# ════════════════════════════════════════════════════════════════

def save_tuning_results(results_df, save_dir=None):
    """Save tuning results to CSV.

    Args:
        results_df: DataFrame summarized from dicts returned by
                    ``run_grid_search`` / ``run_two_stage_search``.
        save_dir: Save directory (default ``TABLES_DIR``).
    """
    if save_dir is None:
        save_dir = TABLES_DIR
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "tuning_results.csv")
    results_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Tuning results saved → {path}")
