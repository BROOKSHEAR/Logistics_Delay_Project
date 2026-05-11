"""
模型比较与假设检验模块。

提供 TimeSeriesSplit 时序交叉验证、Bootstrap AUC 置信区间、
及模型排名分析，用于科学比较多个模型在时序数据上的表现。

用法:
    from logistics_delay.models.comparison import run_comparison
    results = run_comparison(df)
    # results.auc_ci        → DataFrame: 各模型的 AUC 置信区间
    # results.rankings_df   → DataFrame: 各模型的排名分布
    # results.win_matrix    → DataFrame: 配对胜率矩阵
"""
from __future__ import annotations

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
#  最佳超参数（来自 04_tuning.ipynb 调优结果）
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
    },
}

# 使用 XGB 风格特征（含 category dtype）的模型
_N_TREE_MODELS = {"CatBoost", "XGBoost", "LightGBM"}


# ════════════════════════════════════════════════════════════════
#  私有辅助函数
# ════════════════════════════════════════════════════════════════

def _create_model(model_name: str, spw: float):
    """使用最佳参数创建模型实例，并动态注入 scale_pos_weight / class_weights。"""
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
    """根据模型类型选择对应的特征子集并处理 dtype。"""
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
    """使用 sklearn ``TimeSeriesSplit`` 创建时序交叉验证折。

    数据须已按时间排序。保持时间顺序，前 80% → 后 20% 逐折推进。

    Args:
        df: 已排序 DataFrame。
        n_splits: 折数（默认 5）。

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
    """Bootstrap 方法计算 AUC 置信区间。

    对测试集样本有放回重采样 n_resamples 次，
    每次计算 AUC，得到 AUC 的经验分布，取百分位数。

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
        aucs[i] = roc_auc_score(yt[idx], yp[idx])

    return (
        float(np.mean(aucs)),
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
        aucs,
    )


# ════════════════════════════════════════════════════════════════
#  公开 API
# ════════════════════════════════════════════════════════════════

def run_comparison(
    df: pd.DataFrame,
    models: list[str] | None = None,
    n_splits: int = 5,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """运行完整的时间序列 CV 模型比较。

    1. ``TimeSeriesSplit(n_splits=5)`` 创建 5 个时序 fold
    2. 每个 fold 训练全部 6 个模型（使用调优最佳参数）
    3. 用 Bootstrap 计算每个模型的 AUC 置信区间（样本级重采样 2000 次）
    4. 计算跨 fold 的排名分布和配对胜率矩阵

    Args:
        df: 完整 DataFrame（须含 Answer 和 trip_start_date）。
        models: 模型名称列表，默认用 _BEST_PARAMS 中的全部 6 个。
        n_splits: TimeSeriesSplit 折数（默认 5）。
        n_bootstrap: Bootstrap 重采样次数。
        seed: 随机种子。

    Returns:
        dict，包含以下键:
        - ``auc_ci``: DataFrame [model, mean_auc, ci_lower, ci_upper, std_auc]
        - ``rankings_df``: DataFrame [model, rank_1..rank_N, avg_rank]
        - ``win_matrix``: DataFrame (N×N, 行=模型, 值=行模型胜列模型的 fold 比例)
        - ``fold_aucs``: DataFrame [fold, model_1, ..., model_N]
    """
    if models is None:
        models = list(_BEST_PARAMS.keys())

    df_sorted = df.sort_values("trip_start_date").reset_index(drop=True)
    folds = _create_tscv_folds(df_sorted, n_splits=n_splits)
    n_folds = len(folds)
    print(f"[comparison] TimeSeriesSplit({n_splits}) → {n_folds} 个时序 fold")
    print(f"[comparison] 模型: {models}")
    print(f"[comparison] Bootstrap: {n_bootstrap}")

    # ── 存储折中的结果 ──
    fold_auc_point: dict[str, list[float]] = {m: [] for m in models}
    fold_bs_pool: dict[str, list[np.ndarray]] = {m: [] for m in models}

    for train_idx, test_idx, label in folds:
        df_tr = df_sorted.loc[train_idx]
        df_te = df_sorted.loc[test_idx]
        y_train = df_tr["Answer"]
        y_test = df_te["Answer"]
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        spw = n_neg / max(n_pos, 1)

        print(f"\n  [{label}]  train={len(df_tr)}  test={len(df_te)}  "
              f"spw={spw:.4f}")

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
                    y_test, y_prob, n_bootstrap, seed + hash(mname) % 10000,
                )

                fold_auc_point[mname].append(auc_point)
                fold_bs_pool[mname].append(bs_samples)
                print(f"    {mname:<20s}  AUC={auc_point:.4f}  "
                      f"95%CI=[{ci_low:.4f}, {ci_high:.4f}]")
            except Exception as exc:
                fold_auc_point[mname].append(np.nan)
                fold_bs_pool[mname].append(np.array([np.nan]))
                print(f"    {mname:<20s}  ERROR: {exc}")

    # ════════════════════════════════════════════════════════════
    #  汇总
    # ════════════════════════════════════════════════════════════

    # 1. AUC 置信区间
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

    # 2. 排名分布
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

    # 3. 配对胜率矩阵
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

    # ── 打印汇总 ──
    print("\n\n" + "=" * 62)
    print("    模型 AUC 置信区间（Bootstrap 95% CI）")
    print("=" * 62)
    print(f"  {'模型':<22s} {'均值AUC':>8s} {'下限':>8s} {'上限':>8s} {'标准差':>8s}")
    print("  " + "-" * 58)
    for _, r in auc_ci.iterrows():
        print(f"  {r['model']:<22s} {r['mean_auc']:>8.4f} {r['ci_lower']:>8.4f} "
              f"{r['ci_upper']:>8.4f} {r['std_auc']:>8.4f}")

    print("\n\n" + "=" * 62)
    print("    模型排名分布（值 = 夺得该名次的 fold 数）")
    print("=" * 62)
    print(rank_dist.to_string())

    print("\n\n" + "=" * 62)
    print("    配对胜率矩阵（行模型 胜 列模型的 fold 比例）")
    print("=" * 62)
    print(win_mat.to_string(float_format=lambda x: f"{x:.1%}"))

    return {
        "auc_ci":       auc_ci,
        "rankings_df":  rank_dist,
        "win_matrix":   win_mat,
        "fold_aucs":    auc_mat,
    }
