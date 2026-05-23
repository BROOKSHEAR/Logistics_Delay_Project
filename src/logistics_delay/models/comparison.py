"""
Model comparison and hypothesis testing module.

Provides TimeSeriesSplit temporal CV, Bootstrap AUC confidence intervals,
and model ranking analysis for scientific comparison of multiple models on temporal data.

Usage:
    from logistics_delay.models.comparison import run_comparison
    results = run_comparison(df)
    # results.auc_ci        → DataFrame: AUC confidence intervals per model
    # results.rankings_df   → DataFrame: ranking distribution per model
    # results.win_matrix    → DataFrame: pairwise win matrix
"""
from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from logistics_delay.features.engineering import FEATURES_ENC, FEATURES_XGB, XGB_CAT_COLS

__all__ = ["run_comparison"]

RANDOM_STATE = 42

# ════════════════════════════════════════════════════════════════
#  Best hyperparameters (from 04_tuning.ipynb tuning results)
# ════════════════════════════════════════════════════════════════

_BEST_PARAMS = {
    "LogisticRegression": {
        "C": 100, "class_weight": "balanced", "max_iter": 1000,
        "l1_ratio": 0, "solver": "lbfgs", "random_state": RANDOM_STATE,
    },
    "DecisionTree": {
        "criterion": "entropy", "max_depth": 4, "min_samples_leaf": 50,
        "min_samples_split": 10, "class_weight": None, "random_state": RANDOM_STATE,
    },
    "RandomForest": {
        "bootstrap": True, "class_weight": None, "max_depth": 7,
        "max_features": None, "min_samples_leaf": 1,
        "min_samples_split": 10, "n_estimators": 200, "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "XGBoost": {
        "learning_rate": 0.01, "max_depth": 10, "n_estimators": 300,
        "subsample": 0.6, "reg_alpha": 1, "reg_lambda": 0.1,
        "min_child_weight": 3, "gamma": 1, "colsample_bytree": 1.0,
        "random_state": RANDOM_STATE, "verbosity": 0, "enable_categorical": True,
        "n_jobs": -1,
    },
    "LightGBM": {
        "learning_rate": 0.03, "num_leaves": 31, "n_estimators": 100,
        "subsample": 0.6, "max_depth": -1, "min_child_samples": 50,
        "reg_alpha": 0, "reg_lambda": 0.1, "colsample_bytree": 1.0,
        "random_state": RANDOM_STATE, "verbose": -1,
    },
    "CatBoost": {
        "learning_rate": 0.05, "depth": 8, "iterations": 100,
        "l2_leaf_reg": 5, "border_count": 64,
        "bagging_temperature": 1, "random_strength": 0,
        "random_seed": RANDOM_STATE, "verbose": 0,
        "train_dir": tempfile.gettempdir(),
    },
}

# Models using XGB-style features (with category dtype)
_N_TREE_MODELS = {"CatBoost", "XGBoost", "LightGBM"}


# ════════════════════════════════════════════════════════════════
#  Private helper functions
# ════════════════════════════════════════════════════════════════

def _create_model(model_name: str, spw: float):
    """Create model instance with best params, dynamically inject scale_pos_weight / class_weights."""
    params = _BEST_PARAMS[model_name].copy()
    if model_name in ("XGBoost", "LightGBM"):
        params["scale_pos_weight"] = spw
    elif model_name == "CatBoost":
        params["class_weights"] = {0: 1.0, 1: spw}

    factory = {
        "LogisticRegression": LogisticRegression,
        "DecisionTree": DecisionTreeClassifier,
        "RandomForest": RandomForestClassifier,
        "XGBoost": XGBClassifier,
        "LightGBM": LGBMClassifier,
        "CatBoost": CatBoostClassifier,
    }
    return factory[model_name](**params)


def _select_features(model_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Select appropriate feature subset and handle dtype based on model type."""
    if model_name in _N_TREE_MODELS:
        X = df[FEATURES_XGB].copy()
        for c in XGB_CAT_COLS:
            if c in X:
                X[c] = X[c].astype("category")
    else:
        X = df[FEATURES_ENC].copy()
    return X


def _create_tscv_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> list[tuple[pd.Index, pd.Index, str]]:
    """Create temporal CV folds using sklearn ``TimeSeriesSplit``.

    Data must be sorted by time. Maintains temporal order, 80% → 20% per fold.

    Args:
        df: Sorted DataFrame.
        n_splits: Number of folds (default 5).

    Returns:
        [(train_idx, test_idx, label), ...]。
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for i, (tr_idx, te_idx) in enumerate(tscv.split(df)):
        folds.append((
            df.index[tr_idx],
            df.index[te_idx],
            f"fold_{i+1}",
        ))
    return folds


def _bootstrap_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float, np.ndarray]:
    """Bootstrap AUC confidence interval computation.

    Resample test set with replacement n_resamples times,
    compute AUC each time, get empirical distribution, take percentiles.

    Returns:
        (mean_auc, ci_lower, ci_upper, bootstrap_samples)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob)
    aucs = np.empty(n_resamples)

    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(yt[idx])) < 2:
            aucs[i] = np.nan
        else:
            aucs[i] = roc_auc_score(yt[idx], yp[idx])

    return (
        float(np.nanmean(aucs)),
        float(np.nanpercentile(aucs, 2.5)),
        float(np.nanpercentile(aucs, 97.5)),
        aucs,
    )


# ════════════════════════════════════════════════════════════════
#  Public API
# ════════════════════════════════════════════════════════════════

def run_comparison(
    df: pd.DataFrame,
    models: list[str] | None = None,
    n_splits: int = 5,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Run full temporal CV model comparison.

    1. ``TimeSeriesSplit(n_splits=5)`` creates 5 temporal folds
    2. Each fold trains all 6 models (using tuned best params)
    3. Bootstrap computes AUC confidence intervals per model (2000 resamples)
    4. Compute cross-fold ranking distribution and pairwise win matrix

    Args:
        df: Full DataFrame (must contain Answer and trip_start_date).
        models: List of model names, defaults to all 6 from _BEST_PARAMS.
        n_splits: Number of TimeSeriesSplit folds (default 5).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        dict with keys:
        - ``auc_ci``: DataFrame [model, mean_auc, ci_lower, ci_upper, std_auc]
        - ``rankings_df``: DataFrame [model, rank_1..rank_N, avg_rank]
        - ``win_matrix``: DataFrame (N×N, row=model, val=fraction where row beats col)
        - ``fold_aucs``: DataFrame [fold, model_1, ..., model_N]
    """
    if models is None:
        models = list(_BEST_PARAMS.keys())

    df_sorted = df.sort_values("trip_start_date").reset_index(drop=True)
    folds = _create_tscv_folds(df_sorted, n_splits=n_splits)
    n_folds = len(folds)
    print(f"[comparison] TimeSeriesSplit({n_splits}) → {n_folds} temporal folds")
    print(f"[comparison] Models: {models}")
    print(f"[comparison] Bootstrap: {n_bootstrap}")

    # ── Store per-fold results ──
    fold_auc_point: dict[str, list[float]] = {m: [] for m in models}
    fold_bs_pool: dict[str, list[np.ndarray]] = {m: [] for m in models}
    fold_details: list[dict] = []  # per-fold per-model AUC + CI

    for fold_idx, (train_idx, test_idx, label) in enumerate(folds):
        df_tr = df_sorted.loc[train_idx]
        df_te = df_sorted.loc[test_idx]
        y_train = df_tr["Answer"]
        y_test = df_te["Answer"]
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        spw = n_neg / max(n_pos, 1)

        print(f"\n  [{label}]  train={len(df_tr)}  test={len(df_te)}  "
              f"spw={spw:.4f}")

        # All models in the same fold share the same bootstrap seed for fair comparison
        bootstrap_seed = seed + fold_idx

        for mname in models:
            try:
                X_tr = _select_features(mname, df_tr)
                X_te = _select_features(mname, df_te)
                model = _create_model(mname, spw)

                if mname == "CatBoost":
                    model.fit(X_tr, y_train, cat_features=XGB_CAT_COLS)
                elif mname == "LightGBM":
                    model.fit(X_tr, y_train, categorical_feature=XGB_CAT_COLS)
                else:
                    model.fit(X_tr, y_train)

                y_prob = model.predict_proba(X_te)[:, 1]
                auc_point = float(roc_auc_score(y_test, y_prob))
                _, ci_low, ci_high, bs_samples = _bootstrap_auc(
                    y_test, y_prob, n_bootstrap, bootstrap_seed,
                )

                fold_auc_point[mname].append(auc_point)
                fold_bs_pool[mname].append(bs_samples)
                fold_details.append({
                    "fold": label,
                    "model": mname,
                    "auc": auc_point,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                })
                print(f"    {mname:<20s}  AUC={auc_point:.4f}  "
                      f"95%CI=[{ci_low:.4f}, {ci_high:.4f}]")
            except Exception as exc:
                fold_auc_point[mname].append(np.nan)
                fold_bs_pool[mname].append(np.array([np.nan]))
                print(f"    {mname:<20s}  ERROR: {exc}")

    # ════════════════════════════════════════════════════════════
    #  Aggregation
    # ════════════════════════════════════════════════════════════

    # 1. AUC confidence intervals
    ci_rows = []
    for mname in models:
        bs = np.concatenate(fold_bs_pool[mname])
        bs = bs[~np.isnan(bs)]
        if len(bs) == 0:
            continue
        ci_rows.append({
            "model":     mname,
            "mean_auc":  float(np.mean(bs)),
            "ci_lower":  float(np.percentile(bs, 2.5)),
            "ci_upper":  float(np.percentile(bs, 97.5)),
            "std_auc":   float(np.std(bs)),
        })
    auc_ci = (
        pd.DataFrame(ci_rows)
        .sort_values("mean_auc", ascending=False)
        .reset_index(drop=True)
    )

    # 2. Ranking distribution
    auc_mat = pd.DataFrame({m: fold_auc_point[m] for m in models})
    rank_mat = auc_mat.rank(axis=1, ascending=False, method="min").astype(int)
    n_m = len(models)
    rank_cols = [f"rank_{r}" for r in range(1, n_m + 1)]
    rank_dist = pd.DataFrame(0, index=models, columns=rank_cols, dtype=int)
    for mname in models:
        for r in range(1, n_m + 1):
            rank_dist.loc[mname, f"rank_{r}"] = int((rank_mat[mname] == r).sum())
    rank_dist["avg_rank"] = rank_mat.mean().round(2)
    rank_dist = rank_dist.sort_values("avg_rank")

    # 3. Pairwise win matrix
    win_arr = np.full((n_m, n_m), 0.5)
    for i, ma in enumerate(models):
        for j, mb in enumerate(models):
            if i == j:
                continue
            va = np.array(fold_auc_point[ma], dtype=float)
            vb = np.array(fold_auc_point[mb], dtype=float)
            valid = ~(np.isnan(va) | np.isnan(vb))
            if valid.sum() > 0:
                win_arr[i, j] = (va[valid] > vb[valid]).mean()
    win_mat = pd.DataFrame(win_arr, index=models, columns=models)

    # ── Print summary ──
    print("\n\n" + "=" * 62)
    print("    Model AUC confidence intervals (Bootstrap 95% CI)")
    print("=" * 62)
    print(f"  {'Model':<22s} {'Mean AUC':>8s} {'Lower':>8s} {'Upper':>8s} {'Std':>8s}")
    print("  " + "-" * 58)
    for _, r in auc_ci.iterrows():
        print(f"  {r['model']:<22s} {r['mean_auc']:>8.4f} {r['ci_lower']:>8.4f} "
              f"{r['ci_upper']:>8.4f} {r['std_auc']:>8.4f}")

    print("\n\n" + "=" * 62)
    print("    Model ranking distribution (value = folds where model achieved rank)")
    print("=" * 62)
    print(rank_dist.to_string())

    print("\n\n" + "=" * 62)
    print("    Pairwise win matrix (row model beats column model, fraction of folds)")
    print("=" * 62)
    print(win_mat.to_string(float_format=lambda x: f"{x:.1%}"))

    return {
        "auc_ci":       auc_ci,
        "rankings_df":  rank_dist,
        "win_matrix":   win_mat,
        "fold_aucs":    auc_mat,
        "fold_details": pd.DataFrame(fold_details),
    }
