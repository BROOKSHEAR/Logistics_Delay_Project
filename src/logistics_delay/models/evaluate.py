"""
模型评估模块。

提供统一的 ``evaluate`` 和 ``get_model`` 函数，
供训练 notebook 和消融实验重复使用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from logistics_delay.utils.paths import RANDOM_STATE

# ── 支持的模型列表（用于循环） ──
ALL_MODELS = [
    "LogisticRegression", "DecisionTree", "RandomForest",
    "XGBoost", "CatBoost", "LightGBM",
]

# ── CatBoost / LightGBM 使用的原生类别特征名 ──
XGB_CAT_COLS = [
    "vehicleType", "OriginLocation_Code", "DestinationLocation_Code",
    "GpsProvider", "booking_prefix", "origin_city", "dest_city", "customerID",
]


def evaluate(model, X_test, y_test, model_name: str = "") -> dict:
    """统一评估函数。

    Returns:
        dict 包含 ``auc``, ``f1``, ``accuracy``, ``model``。
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "model": model_name,
        "auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
    }


def get_model(model_name: str,
              spw: float | None = None,
              cat_features: list | None = None):
    """按名称返回与主实验参数一致的模型实例。

    Args:
        model_name: 模型名称 (``ALL_MODELS`` 之一)。
        spw: ``scale_pos_weight`` 或 ``class_weights`` 的参考值。
        cat_features: CatBoost 的 ``cat_features`` 参数。

    Returns:
        未拟合的模型实例。
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced",
        ),
        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=8, random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, scale_pos_weight=spw,
            random_state=RANDOM_STATE, verbosity=0, enable_categorical=True,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            class_weights={0: 1.0, 1: spw} if spw is not None else None,
            random_seed=RANDOM_STATE, verbose=0,
            cat_features=cat_features,
        ),
        "LightGBM": None,  # LightGBM 参数因版本差异较大，单独处理
    }
    if model_name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=200, scale_pos_weight=spw,
            random_state=RANDOM_STATE, verbose=-1,
        )
    return models[model_name]


def _prepare_xgb_input(model_name: str,
                       X_enc: pd.DataFrame,
                       X_xgb: pd.DataFrame) -> pd.DataFrame:
    """根据不同模型类型返回对应的特征矩阵。

    三个树模型（CatBoost / XGBoost / LightGBM）都原生支持
    pandas ``category`` dtype，直接返回 ``X_xgb``。
    其余 sklearn 模型返回 ``X_enc``。
    """
    if model_name in ("CatBoost", "XGBoost", "LightGBM"):
        return X_xgb
    return X_enc
