"""
Model evaluation module.

Provides unified ``evaluate`` and ``get_model`` functions,
reused by training notebooks and ablation experiments.
"""
from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from logistics_delay.utils.paths import RANDOM_STATE

# ── Supported model list (for iteration) ──
ALL_MODELS = [
    "LogisticRegression", "DecisionTree", "RandomForest",
    "XGBoost", "CatBoost", "LightGBM",
]

# ── Raw categorical feature names for CatBoost / LightGBM ──
XGB_CAT_COLS = [
    "vehicleType", "OriginLocation_Code", "DestinationLocation_Code",
    "GpsProvider", "booking_prefix", "origin_city", "dest_city", "customerID",
]


def evaluate(model, X_test, y_test, model_name: str = "") -> dict:
    """Unified evaluation function。

    Returns:
        dict containing ``auc``, ``f1``, ``accuracy``, ``model``.
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
    """Return a model instance consistent with the main experiment by name.

    Args:
        model_name: Model name (one of ``ALL_MODELS``).
        spw: Reference value for ``scale_pos_weight`` or ``class_weights``.
        cat_features: CatBoost ``cat_features`` parameter.

    Returns:
        Unfitted model instance.
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
            train_dir=tempfile.gettempdir(),
            cat_features=cat_features,
        ),
        "LightGBM": None,  # LightGBM params vary significantly by version; handled separately
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
    """Return the appropriate feature matrix based on model type.

    The three tree models (CatBoost / XGBoost / LightGBM) natively support
    pandas ``category`` dtype, so ``X_xgb`` is returned directly.
    Other sklearn models return ``X_enc``.
    """
    if model_name in ("CatBoost", "XGBoost", "LightGBM"):
        return X_xgb
    return X_enc
