"""
Feature ablation experiment script.

Provides data loading, leave-one-out feature ablation, learning curve analysis, and result saving.
Uses temporal split only: sort by ``trip_start_date`` then 80/20 split.

Usage:
    python -m src.logistics_delay.ablation.feature_ablation
"""
from __future__ import annotations

import os
import tempfile
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

from logistics_delay.utils.paths import DATA_RAW, SEED, TABLES_DIR
from logistics_delay.features.engineering import FEATURES_XGB, XGB_CAT_COLS
from logistics_delay.features.distance_fill_geo import DistanceFiller

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
#  1. Data loading and feature engineering
# ════════════════════════════════════════════════════════════════

def load_and_prepare_data() -> pd.DataFrame:
    """Load feature data from preprocessed file (skip raw → engineer_features pipeline).

    Reads ``data/processed/truck_delay_handled_file.xlsx``,
    sorts by ``trip_start_date`` chronologically.

    Returns:
        Sorted full DataFrame with ``Answer`` column and all 61 feature columns.
    """
    from logistics_delay.data.loader import load_processed
    df = load_processed()
    df_sorted = df.sort_values("trip_start_date").reset_index(drop=True)
    print(f"[data] Final shape: {df_sorted.shape}")
    print(f"[data] Positive rate: {df_sorted['Answer'].mean():.4f}")
    return df_sorted


# ════════════════════════════════════════════════════════════════
#  2. Feature ablation (Leave-One-Out)
# ════════════════════════════════════════════════════════════════

_DEFAULT_CB_PARAMS: dict = {
    "iterations": 200,
}


def _build_cb_model(
    spw: float,
    cat_features: list[str] | None,
    extra_params: dict | None = None,
) -> CatBoostClassifier:
    """Build CatBoostClassifier merging default + tuned + dynamic parameters."""
    params = dict(_DEFAULT_CB_PARAMS)
    if extra_params:
        params.update(extra_params)
    params.update(
        class_weights={0: 1.0, 1: spw},
        random_seed=SEED,
        verbose=0,
        train_dir=tempfile.gettempdir(),
        cat_features=cat_features if cat_features else None,
    )
    return CatBoostClassifier(**params)


def run_feature_ablation(
    df_sorted: pd.DataFrame,
    feature_list: list[str] | None = None,
    catboost_params: dict | None = None,
) -> pd.DataFrame:
    """Leave-one-out feature ablation using CatBoost.

    After temporal 80/20 split by ``trip_start_date``,
    First train on all features as baseline, then remove one feature at a time,
    Record AUC and F1 changes. Pass categorical features via ``cat_features``.

    Args:
        df_sorted: DataFrame sorted by time from ``load_and_prepare_data``.
        feature_list: Features to ablate (default ``FEATURES_XGB``).

    Returns:
        Ablation result DataFrame, columns:
        - ``removed_feat``: Removed feature name (first row is ``(Full Features)``)
        - ``auc``: Validation AUC
        - ``f1``: Validation F1
        - ``auc_drop``: ``Full AUC - Ablated AUC`` (pp).
          Positive = feature removal degraded performance (feature important),
          Negative = feature removal improved performance (feature noisy).
        - ``f1_drop``: ``Full F1 - Ablated F1`` (pp), same sign convention.
    """
    if feature_list is None:
        feature_list = FEATURES_XGB

    print("\n" + "=" * 50)
    print("  Feature ablation (CatBoost Leave-One-Out)")
    print("=" * 50)

    # ── Temporal split 80/20 ──
    split_idx = int(len(df_sorted) * 0.8)
    X_train = df_sorted.loc[: split_idx - 1, feature_list].reset_index(drop=True)
    X_test = df_sorted.loc[split_idx:, feature_list].reset_index(drop=True)
    y_train = df_sorted.loc[: split_idx - 1, "Answer"].reset_index(drop=True)
    y_test = df_sorted.loc[split_idx:, "Answer"].reset_index(drop=True)
    spw = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    print(f"[split] Train: {len(y_train)} / Test: {len(y_test)}, spw={spw:.4f}")

    # ── Full-feature baseline model ──
    cat_feats_full = [c for c in XGB_CAT_COLS if c in feature_list]
    model_full = _build_cb_model(spw, cat_feats_full, catboost_params)
    model_full.fit(X_train, y_train)
    y_pred_full = model_full.predict(X_test)
    y_prob_full = model_full.predict_proba(X_test)[:, 1]
    full_auc = roc_auc_score(y_test, y_prob_full)
    full_f1 = f1_score(y_test, y_pred_full)
    print(f"[full] AUC = {full_auc * 100:.2f}%  F1 = {full_f1:.4f}")

    # ── Leave-one-out ablation ──
    results = []
    for feat in feature_list:
        subset = [f for f in feature_list if f != feat]
        cat_feats_sub = [c for c in cat_feats_full if c != feat]

        model = _build_cb_model(spw, cat_feats_sub, catboost_params)
        model.fit(X_train[subset], y_train)
        y_pred = model.predict(X_test[subset])
        y_prob = model.predict_proba(X_test[subset])[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        auc_drop_val = round((full_auc - auc) * 100, 4)
        f1_drop_val = round((full_f1 - f1) * 100, 4)
        results.append({
            "removed_feat": feat,
            "auc": round(auc, 6),
            "f1": round(f1, 6),
            "auc_drop": auc_drop_val,
            "f1_drop": f1_drop_val,
        })
        print(f"  Remove {feat:<30s}  AUC = {auc * 100:.2f}%  "
              f"(drop={auc_drop_val:+.2f}pp)  "
              f"F1 = {f1:.4f}  (drop={f1_drop_val:+.4f}pp)")

    # Insert full-feature baseline row
    full_row = pd.DataFrame([{
        "removed_feat": "(Full Features)",
        "auc": round(full_auc, 6),
        "f1": round(full_f1, 6),
        "auc_drop": 0.0,
        "f1_drop": 0.0,
    }])
    result_df = pd.concat([full_row, pd.DataFrame(results)], ignore_index=True)
    return result_df


# ════════════════════════════════════════════════════════════════
#  3. Geographic ablation
# ════════════════════════════════════════════════════════════════

def run_geo_ablation(
    df_sorted: pd.DataFrame,
    catboost_params: dict | None = None,
) -> pd.DataFrame:
    """Geo ablation: compare three distance imputation strategies on CatBoost.

    All three strategies share identical features and hyperparams, only varying
    how missing ``TRANSPORTATION_DISTANCE_IN_KM`` values are filled:
      - **Geo**: ``DistanceFiller`` geographic proximity + median fallback.
      - **Median**: All missing values filled with median.
      - **Mean**: All missing values filled with mean.

    Args:
        df_sorted: DataFrame sorted by time from ``load_and_prepare_data``.
           The ``_dist_original`` column (saved during preprocessing by ``cleaner.fill_distance_geo``)
           stores original distances (with NaN) to identify originally missing rows.

    Returns:
        Ablation result DataFrame, columns:
        - ``strategy``: Strategy name
        - ``auc``: Validation AUC
        - ``f1``: Validation F1
    """
    print("\n" + "=" * 50)
    print("  Geo ablation experiment")
    print("=" * 50)

    # ── Extract original distance info ──
    missing_mask = df_sorted["_dist_original"].isna()
    orig_dist = df_sorted["_dist_original"].values
    non_missing = orig_dist[~missing_mask.values]
    median_val = np.median(non_missing)
    mean_val = np.mean(non_missing)
    print(f"  Missing distance: {missing_mask.sum()} / {len(orig_dist)}")
    print(f"  Median: {median_val:.2f}, Mean: {mean_val:.2f}")

    # ── Three strategies ──
    strategies: dict[str, float | None] = {
        "geo": None,
        "median": median_val,
        "mean": mean_val,
    }
    strategy_labels = {
        "geo": "Geo",
        "median": "Median",
        "mean": "Mean",
    }

    feature_list = FEATURES_XGB
    cat_feats = [c for c in XGB_CAT_COLS if c in feature_list]
    split_idx = int(len(df_sorted) * 0.8)

    results = []
    for key, fill_value in strategies.items():
        df_copy = df_sorted.copy()

        if fill_value is None:
            # Geo fill: use existing geo + median fallback values from df_sorted
            pass
        else:
            # Median / Mean fill
            dist_col = orig_dist.copy()
            dist_col[missing_mask.values] = fill_value
            df_copy["TRANSPORTATION_DISTANCE_IN_KM"] = dist_col

        X_tr = df_copy.loc[: split_idx - 1, feature_list].reset_index(drop=True)
        X_te = df_copy.loc[split_idx:, feature_list].reset_index(drop=True)
        y_tr = df_copy.loc[: split_idx - 1, "Answer"].reset_index(drop=True)
        y_te = df_copy.loc[split_idx:, "Answer"].reset_index(drop=True)
        spw = len(y_tr[y_tr == 0]) / max(len(y_tr[y_tr == 1]), 1)

        model = _build_cb_model(spw, cat_feats, catboost_params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
        f1 = f1_score(y_te, y_pred)

        results.append({
            "strategy": strategy_labels[key],
            "auc": round(auc, 6),
            "f1": round(f1, 6),
        })
        print(f"  {strategy_labels[key]:<20s}  AUC={auc * 100:.2f}%  F1={f1:.4f}")

    return pd.DataFrame(results)


def save_geo_results(
    geo_df: pd.DataFrame,
    save_dir: str | os.PathLike | None = None,
) -> None:
    """Save geo ablation results to CSV.

    Args:
        geo_df: DataFrame returned by ``run_geo_ablation``.
        save_dir: Save directory (default ``TABLES_DIR``).
    """
    if save_dir is None:
        save_dir = TABLES_DIR
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "geo_ablation_results.csv")
    geo_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] Geo ablation results → {path}")


# ════════════════════════════════════════════════════════════════
#  4. Cumulative feature learning curves
# ════════════════════════════════════════════════════════════════

def run_learning_curves(
    df_sorted: pd.DataFrame,
    feature_list: list[str] | None = None,
    ablation_df: pd.DataFrame | None = None,
    catboost_params: dict | None = None,
) -> pd.DataFrame:
    """Cumulative feature learning curves.

    First rank features by ``auc_drop`` descending via ``run_feature_ablation``,
    then cumulatively add features from most to least important, training CatBoost
    with temporal split at each step, recording AUC.

    Args:
        df_sorted: DataFrame sorted by time from ``load_and_prepare_data``.
        feature_list: Feature list (default ``FEATURES_XGB``).

    Returns:
        Learning curve DataFrame, columns:
        - ``feature_added``: Feature added this round
        - ``n_features``: Current feature count
        - ``auc``: Validation AUC
    """
    if feature_list is None:
        feature_list = FEATURES_XGB

    print("\n" + "=" * 50)
    print("  Cumulative feature learning curves")
    print("=" * 50)

    # ── 1. Get feature importance ranking from ablation ──
    if ablation_df is None:
        ablation_df = run_feature_ablation(df_sorted, feature_list)
    imp_df = ablation_df[ablation_df["removed_feat"] != "(Full Features)"].copy()
    imp_df = imp_df.sort_values("auc_drop", ascending=False)
    feature_order = imp_df["removed_feat"].tolist()
    print(f"\n  Feature addition order (by auc_drop descending):")
    for i, f in enumerate(feature_order, 1):
        print(f"    {i:>2d}. {f}")

    # ── 2. Temporal split ──
    split_idx = int(len(df_sorted) * 0.8)
    y_train = df_sorted.loc[: split_idx - 1, "Answer"].reset_index(drop=True)
    y_test = df_sorted.loc[split_idx:, "Answer"].reset_index(drop=True)
    cat_feats_all = [c for c in XGB_CAT_COLS if c in feature_list]

    print(f"\n[split] Train: {split_idx} / Test: {len(df_sorted) - split_idx}")

    # ── 3. Cumulative feature addition ──
    records = []
    cumulative: list[str] = []
    for feat in feature_order:
        cumulative.append(feat)
        X_tr = df_sorted.loc[: split_idx - 1, cumulative].reset_index(drop=True)
        X_te = df_sorted.loc[split_idx:, cumulative].reset_index(drop=True)
        cat_feats = [c for c in cat_feats_all if c in cumulative]

        spw = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
        model = _build_cb_model(spw, cat_feats, catboost_params)
        model.fit(X_tr, y_train)
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        records.append({
            "feature_added": feat,
            "n_features": len(cumulative),
            "auc": round(auc, 6),
        })
        print(f"  +{feat:<30s}  n={len(cumulative):>2d}  AUC={auc * 100:.2f}%")

    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════
#  5. Result saving
# ════════════════════════════════════════════════════════════════

def save_results(
    ablation_df: pd.DataFrame,
    lc_df: pd.DataFrame,
    save_dir: str | os.PathLike | None = None,
) -> None:
    """Save feature ablation and learning curve results to CSV.

    Args:
        ablation_df: DataFrame from ``run_feature_ablation``.
            ``auc_drop`` = Full AUC - Ablated AUC (positive = degradation, negative = improvement).
        lc_df: DataFrame from ``run_learning_curves``.
            Contains ``feature_added``, ``n_features``, ``auc`` columns.
        save_dir: Save directory (default ``TABLES_DIR``).
    """
    if save_dir is None:
        save_dir = TABLES_DIR

    os.makedirs(save_dir, exist_ok=True)

    ablation_path = os.path.join(save_dir, "feature_ablation_results.csv")
    lc_path = os.path.join(save_dir, "learning_curves_results.csv")

    ablation_df.to_csv(ablation_path, index=False, encoding="utf-8-sig")
    lc_df.to_csv(lc_path, index=False, encoding="utf-8-sig")

    print(f"\n[OK] Feature ablation results → {ablation_path}")
    print(f"[OK] Learning curve results → {lc_path}")


# ════════════════════════════════════════════════════════════════
#  Main entry
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Ablation script — main entry")
    print("=" * 60)

    df_sorted = load_and_prepare_data()

    ablation_results = run_feature_ablation(df_sorted)
    lc_results = run_learning_curves(df_sorted, ablation_df=ablation_results)
    save_results(ablation_results, lc_results)
    print(df_sorted.columns.tolist())
    geo_results = run_geo_ablation(df_sorted)
    save_geo_results(geo_results)

    print("\n" + "=" * 60)
    print("  All complete")
    print("=" * 60)
