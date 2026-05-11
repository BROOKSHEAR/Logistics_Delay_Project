"""
两阶段超参数调优模块。

第一阶段: LogisticRegression + DecisionTree 用 GridSearchCV + TimeSeriesSplit 穷举。
第二阶段: RandomForest + XGBoost + LightGBM + CatBoost 用 RandomizedSearchCV 粗搜
          + 精细网格 GridSearchCV 精搜 (3 关键参数 × 3 候选 = ≤27 组合)。

用法:
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
#  参数空间定义
# ════════════════════════════════════════════════════════════════

# ── 逻辑回归: 三个 param_grid 避免 penalty/solver 非法组合 ──

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

# ── 决策树 ──

dt_params = {
    "max_depth": [3, 4, 5, 6, 8],
    "min_samples_split": [10, 20, 40],
    "min_samples_leaf": [5, 10, 20, 50],
    "criterion": ["gini", "entropy"],
    "class_weight": ["balanced", None],
}

# ── 随机森林（完整参数空间）──

rf_params = {
    "n_estimators": [50, 100, 200, 300, 500],
    "max_depth": [3, 5, 7, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
    "class_weight": ["balanced", "balanced_subsample", None],
}

# ── XGBoost（完整参数空间, scale_pos_weight 由 spw_t 动态注入）──

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

# ── LightGBM（完整参数空间）──

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

# ── CatBoost（完整参数空间）──

cb_params = {
    "learning_rate": [0.03, 0.05, 0.1],
    "depth": [4, 6, 8],
    "iterations": [100, 200],
    "l2_leaf_reg": [1, 3, 5],
    "border_count": [64, 128],
    "bagging_temperature": [0, 0.5, 1],
    "random_strength": [0, 0.5, 1],
}

# ── 第二阶段精搜的关键参数（每个模型最重要的 3 个）──

RF_KEY_PARAMS = ["n_estimators", "max_depth", "min_samples_leaf"]
XGB_KEY_PARAMS = ["learning_rate", "max_depth", "n_estimators"]
LGBM_KEY_PARAMS = ["learning_rate", "num_leaves", "n_estimators"]
CB_KEY_PARAMS = ["learning_rate", "depth", "iterations"]


# ════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════

def compute_and_print_spw(y_train, n_splits=5):
    """计算 spw 并打印训练集样本分布和 TimeSeriesSplit 各 fold 延误率。

    Args:
        y_train: 训练集标签 (pd.Series 或 array-like)。
        n_splits: TimeSeriesSplit 折数。

    Returns:
        (spw_t, spw_candidates) 元组:
        - spw_t: 负样本数 / 正样本数
        - spw_candidates: [spw_t*0.5, 0.75, 1.0, 1.25, 1.5]
    """
    print("=" * 60)
    print("  样本分布与 scale_pos_weight 计算")
    print("=" * 60)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    total = len(y_train)
    spw_t = n_neg / n_pos if n_pos > 0 else 1.0

    print(f"\n训练集样本分布:")
    print(f"  总样本: {total}")
    print(f"  正样本 (延误): {n_pos} ({n_pos / total * 100:.2f}%)")
    print(f"  负样本 (准时): {n_neg} ({n_neg / total * 100:.2f}%)")
    print(f"  spw_t (负/正) = {spw_t:.4f}")

    print(f"\nTimeSeriesSplit({n_splits} folds) 各 fold 延误率:")
    print(f"  {'Fold':<6} {'训练延误率':<14} {'验证延误率':<14} {'验证正样本比'}")
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
    print(f"\n动态生成的 scale_pos_weight 候选值:")
    print(f"  {', '.join(f'{v:.4f}' for v in spw_candidates)}")
    print(f"  生成规则: [spw_t×0.5, spw_t×0.75, spw_t×1.0, spw_t×1.25, spw_t×1.5]")

    return spw_t, spw_candidates


def refine_grid(best_params, param_dist, key_params):
    """根据粗搜 best_params 生成精细网格。

    对 ``key_params`` 中的每个参数，从 ``param_dist`` 原始候选值中
    取最优值及其左右相邻各一个（共 ≤3 个值）；边界时向有值方向扩展。
    非关键参数全部固定为 ``best_params`` 中的最优值。

    Args:
        best_params: 粗搜输出的 ``best_params_`` (dict)。
        param_dist: 粗搜参数分布 (dict, 值须为 list-like)。
        key_params: 需要精细搜索的关键参数名列表 (3 个)。

    Returns:
        dict, 可作为 ``GridSearchCV`` 的 ``param_grid``。
    """
    fine_grid = {}

    for param in key_params:
        candidates = list(param_dist.get(param, []))
        if not candidates:
            fine_grid[param] = [best_params[param]]
            continue

        best_val = best_params[param]
        # 找到最优值在候选列表中的位置
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

    # 非关键参数固定为最优值
    for param, val in best_params.items():
        if param not in key_params:
            fine_grid[param] = [val]

    return fine_grid


# ════════════════════════════════════════════════════════════════
#  第一阶段: GridSearchCV 穷举
# ════════════════════════════════════════════════════════════════

def run_grid_search(model, param_grid, X_train, y_train, X_test, y_test):
    """``GridSearchCV`` + ``TimeSeriesSplit(n_splits=5)`` 穷举搜索。

    Args:
        model: 未拟合的 sklearn 模型实例。
        param_grid: ``GridSearchCV`` 的 ``param_grid`` (dict 或 list of dicts)。
        X_train: 训练特征。
        y_train: 训练标签。
        X_test: 测试特征。
        y_test: 测试标签。

    Returns:
        dict, 含 ``model``, ``cv_auc``, ``test_auc``, ``test_f1``, ``best_params``。
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
#  第二阶段: RandomizedSearchCV 粗搜 → GridSearchCV 精搜
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
    """两阶段搜索: ``RandomizedSearchCV`` 粗搜 + 精细网格 ``GridSearchCV`` 精搜。

    精搜只对 ``key_params`` 中的 3 个参数在最优值相邻档位搜索，
    其余参数全部固定为粗搜的 ``best_params_``，总组合数 ≤ 27。

    Args:
        model: 未拟合的模型实例。
        param_dist: 粗搜参数分布 (dict, 值须为 list-like)。
        key_params: 精搜的 3 个关键参数名列表。
        n_iter: ``RandomizedSearchCV`` 采样次数。
        X_train: 训练特征。
        y_train: 训练标签。
        X_test: 测试特征。
        y_test: 测试标签。
        fit_params: 传给 ``.fit()`` 的额外参数 (如 CatBoost 的 ``cat_features``)。
        seed: 随机种子。

    Returns:
        dict, 含 ``model``, ``cv_auc``, ``test_auc``, ``test_f1``, ``best_params``。
    """
    model_name = type(model).__name__
    tscv = TimeSeriesSplit(n_splits=5)

    # ── Stage 1: 粗搜 ──
    print(f"\n{'=' * 60}")
    print(f"  阶段 1/2 — 随机搜索: {model_name} (n_iter={n_iter})")
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
    print(f"  [粗搜] 最佳 CV AUC = {rs.best_score_ * 100:.2f}%")
    for k, v in rs.best_params_.items():
        print(f"    {k}: {v}")

    # ── Stage 2: 精搜 ──
    print(f"\n{'=' * 60}")
    print(f"  阶段 2/2 — 精细网格搜索: {model_name}")
    print(f"{'=' * 60}")

    fine_grid = refine_grid(rs.best_params_, param_dist, key_params)
    total_combos = 1
    for k in key_params:
        total_combos *= len(fine_grid[k])
    print(f"  精细网格 {len(key_params)} 关键参数 × {total_combos} 种组合:")
    for k in key_params:
        print(f"    {k}: {fine_grid[k]}")
    for k, v in fine_grid.items():
        if k not in key_params:
            print(f"    {k}: 固定为 {v[0]}")

    stage2_model = clone(rs.best_estimator_)
    gs = GridSearchCV(
        stage2_model,
        fine_grid,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    gs.fit(X_train, y_train, **(fit_params or {}))
    print(f"  [精搜] 最佳 CV AUC = {gs.best_score_ * 100:.2f}%")

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

    print(f"\n  ✅ {model_name} 完成")
    print(f"     CV AUC  = {result['cv_auc'] * 100:.2f}%")
    print(f"     Test AUC = {result['test_auc'] * 100:.2f}%")
    print(f"     Test F1  = {result['test_f1']:.4f}")

    return result


# ════════════════════════════════════════════════════════════════
#  结果保存
# ════════════════════════════════════════════════════════════════

def save_tuning_results(results_df, save_dir=None):
    """将调参结果保存为 CSV。

    Args:
        results_df: ``run_grid_search`` / ``run_two_stage_search``
                    返回的 dict 汇总成的 DataFrame。
        save_dir: 保存目录 (默认 ``TABLES_DIR``)。
    """
    if save_dir is None:
        save_dir = TABLES_DIR
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "tuning_results.csv")
    results_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 调参结果 → {path}")
