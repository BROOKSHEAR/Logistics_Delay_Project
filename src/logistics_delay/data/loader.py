"""
数据加载模块。

统一从 utils.paths 读取数据路径，提供加载原始数据的函数。
"""
from __future__ import annotations

import pandas as pd

from logistics_delay.utils.paths import check_data_exists, DATA_PROCESSED
from logistics_delay.features.engineering import XGB_CAT_COLS


def load_raw_data() -> pd.DataFrame:
    """从 data/raw/ 加载原始 Excel 数据，不做任何处理。

    Returns:
        原始 DataFrame。
    """
    path = check_data_exists()
    df = pd.read_excel(path)
    print(f"[loader] 原始数据加载完成: {df.shape}")
    return df


def load_raw_data_with_target() -> pd.DataFrame:
    """加载原始数据并构建目标变量 ``Answer`` 列。

    ``Answer`` = 0（ontime == 'G'）| 1（其余，即延误）。

    Returns:
        含 ``Answer`` 列的 DataFrame。
    """
    df = load_raw_data()
    df["Answer"] = df["ontime"].apply(lambda x: 0 if x == "G" else 1)
    counts = df["Answer"].value_counts()
    print(f"[loader] 目标变量构建完成: 延误={counts.get(1, 0)}, 准时={counts.get(0, 0)}")
    print(f"[loader] 延误率: {counts.get(1, 0) / len(df) * 100:.2f}%")
    return df


def load_processed() -> pd.DataFrame:
    """加载已预处理完毕的特征数据 (跳过 raw → engineer_features 流程)。

    读取 ``data/processed/truck_delay_handled_file.xlsx``，
    自动为树模型所需的类别列设置 ``category`` dtype。

    Returns:
        含 ``Answer`` 列和全部特征的 DataFrame (6854×61)。
    """
    path = DATA_PROCESSED / "truck_delay_handled_file.xlsx"
    df = pd.read_excel(path)

    # Excel 不保留 category dtype，手动恢复
    for col in XGB_CAT_COLS:
        if col in df.columns:
            # 填充 NaN + 统一转 str: 避免 XGBoost 因混合类型 (int/str) 报错
            # 以及 CatBoost 因 NaN 类别特征报错
            df[col] = df[col].fillna("UNKNOWN").astype(str).astype("category")

    print(f"[loader] 预处理数据加载完成: {df.shape}")
    print(f"[loader] 延误率: {df['Answer'].mean() * 100:.2f}%")
    return df
