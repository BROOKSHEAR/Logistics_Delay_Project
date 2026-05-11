"""
模型训练模块。

提供数据划分函数和单模型/多模型训练封装。
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from logistics_delay.utils.paths import RANDOM_STATE
from logistics_delay.features.engineering import (
    FEATURES_ENC, FEATURES_XGB, get_feature_lists,
)


# ──────────────── 数据划分函数 ────────────────

def random_split(df: pd.DataFrame,
                 feature_set: str = "enc",
                 test_size: float = 0.2,
                 stratify: bool = True):
    """随机划分（80/20 分层抽样）。

    Args:
        df: 含 ``Answer`` 列和特征列的完整 DataFrame。
        feature_set: ``"enc"`` 或 ``"xgb"``。
        test_size: 测试集比例。
        stratify: 是否按目标变量分层抽样。

    Returns:
        (X_train, X_test, y_train, y_test, scale_pos_weight) 或
        (X_train, X_test, y_train, y_test, scale_pos_weight, X_train_xgb, X_test_xgb)
        若 feature_set="both"。
    """
    if feature_set == "xgb":
        features = FEATURES_XGB
    else:
        features = FEATURES_ENC

    X = df[features]
    y = df["Answer"]

    stratify_y = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify_y,
    )
    spw = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    print(f"[train] 随机划分: 训练 {len(y_train)} / 测试 {len(y_test)}")
    print(f"[train] scale_pos_weight = {spw:.4f}")

    return X_train, X_test, y_train, y_test, spw


def temporal_split(df: pd.DataFrame,
                   feature_set: str = "enc",
                   test_size: float = 0.2):
    """时序划分：按 ``trip_start_date`` 排序，前 ``(1-test_size)`` 训练，后 ``test_size`` 测试。

    确保训练集全部早于测试集，避免数据泄露。

    Args:
        df: 含 ``Answer``、``trip_start_date`` 和特征列的 DataFrame。
        feature_set: ``"enc"`` 或 ``"xgb"``。

    Returns:
        (X_train, X_test, y_train, y_test, scale_pos_weight, split_date)
    """
    if feature_set == "xgb":
        features = FEATURES_XGB
    else:
        features = FEATURES_ENC

    df_sorted = df.sort_values("trip_start_date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    cutoff_date = df_sorted.iloc[split_idx]["trip_start_date"]

    X_train = df_sorted.loc[: split_idx - 1, features].reset_index(drop=True)
    X_test = df_sorted.loc[split_idx:, features].reset_index(drop=True)
    y_train = df_sorted.loc[: split_idx - 1, "Answer"].reset_index(drop=True)
    y_test = df_sorted.loc[split_idx:, "Answer"].reset_index(drop=True)
    spw = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    print(f"[train] 时序划分: 分割点 {cutoff_date.date()}")
    print(f"[train] 训练: {len(y_train)} 条, "
          f"{df_sorted.loc[:split_idx - 1, 'trip_start_date'].min().date()} ~ "
          f"{df_sorted.loc[:split_idx - 1, 'trip_start_date'].max().date()}")
    print(f"[train] 测试: {len(y_test)} 条, "
          f"{df_sorted.loc[split_idx, 'trip_start_date'].date()} ~ "
          f"{df_sorted['trip_start_date'].max().date()}")
    print(f"[train] scale_pos_weight = {spw:.4f}")

    return X_train, X_test, y_train, y_test, spw, cutoff_date


# ──────────────── 训练封装 ────────────────

def train_model(model, X_train, y_train, **fit_kwargs):
    """训练单个模型。

    Args:
        model: sklearn-compatible 模型实例。
        X_train: 训练特征。
        y_train: 训练标签。

    Returns:
        已拟合的模型。
    """
    model.fit(X_train, y_train, **fit_kwargs)
    return model
