"""
特征工程模块。

提供独立的时间特征、分箱、编码函数，以及 ``engineer_features`` 一键管道。

用法:
    from logistics_delay.features.engineering import engineer_features, get_feature_lists
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from logistics_delay.data.cleaner import clean_data, fill_distance_geo

# ──────────────── 特征列表常量（全局统一） ────────────────

XGB_CAT_COLS = [
    "vehicleType", "OriginLocation_Code", "DestinationLocation_Code",
    "GpsProvider", "booking_prefix", "origin_city", "dest_city", "customerID",
]

FEATURES_ENC = [
    "TRANSPORTATION_DISTANCE_IN_KM", "vehicleType_enc", "is_market",
    "start_weekday", "start_month",
    "OriginLocation_Code_enc", "DestinationLocation_Code_enc",
    "min_kms_bin", "planned_days_enc", "booking_prefix_enc",
    "origin_city_enc", "dest_city_enc", "customerID_enc",
    "supplier_is_large", "GpsProvider_enc",
]

FEATURES_XGB = [
    "TRANSPORTATION_DISTANCE_IN_KM", "is_market",
    "start_weekday", "start_month",
    "min_kms_bin", "planned_days_enc",
    "supplier_is_large",
] + XGB_CAT_COLS


def get_feature_lists():
    """返回三个特征列表常量。

    Returns:
        (features_enc, features_xgb, xgb_cat_cols) 元组。
    """
    return FEATURES_ENC, FEATURES_XGB, XGB_CAT_COLS


# ──────────────── 底层独立函数 ────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加 ``start_weekday``（0=周一）和 ``start_month``（1=1月）。"""
    df = df.copy()
    df["start_weekday"] = df["trip_start_date"].dt.dayofweek
    df["start_month"] = df["trip_start_date"].dt.month
    return df


def add_min_kms_bin(df: pd.DataFrame) -> pd.DataFrame:
    """对 ``Minimum_kms_to_be_covered_in_a_day`` 分箱。

    分箱方案: [-1, 200] → 0, (200, 250] → 1, (250, 999] → 2。
    """
    df = df.copy()
    df["min_kms_bin"] = pd.cut(
        df["Minimum_kms_to_be_covered_in_a_day"],
        bins=[-1, 200, 250, 999],
        labels=[0, 1, 2],
    ).astype(float)
    return df


def add_planned_days_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算 ``planned_days`` 及其分箱编码 ``planned_days_enc``。

    从 ``Planned_ETA`` 和 ``BookingID_Date`` 计算计划天数，
    去除负值和 >180 天的异常值后用中位数填充。
    分箱方案: [-1, 1] → 0, (1, 3] → 1, (3, 7] → 2, (7, 999] → 3。
    """
    df = df.copy()
    df["planned_eta_time"] = pd.to_datetime(df["Planned_ETA"], errors="coerce")
    df["planned_days"] = (
        df["planned_eta_time"].dt.normalize()
        - df["BookingID_Date"].dt.normalize()
    ).dt.days

    df.loc[df["planned_days"] < 0, "planned_days"] = None
    df.loc[df["planned_days"] > 180, "planned_days"] = None
    median_val = df["planned_days"].median()
    df["planned_days"] = df["planned_days"].fillna(median_val).astype("Int64")

    df["planned_days_enc"] = pd.cut(
        df["planned_days"],
        bins=[-1, 1, 3, 7, 999],
        labels=[0, 1, 2, 3],
    ).astype(float)
    return df


def add_business_flags(df: pd.DataFrame) -> pd.DataFrame:
    """添加业务标记特征 ``is_market`` 和 ``supplier_is_large``。"""
    df = df.copy()
    df["is_market"] = (df["Market/Regular"] == "Market").astype(int)
    has_alpha = df["supplierID"].str.contains("[A-Za-z]", regex=True, na=False)
    df["supplier_is_large"] = has_alpha.astype(int)
    return df


def encode_label_features(df: pd.DataFrame) -> pd.DataFrame:
    """对所有类别特征执行 LabelEncoder 编码，生成 ``*_enc`` 列。"""
    df = df.copy()
    le = LabelEncoder()

    for col in ["vehicleType", "OriginLocation_Code", "DestinationLocation_Code"]:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    df["GpsProvider_enc"] = le.fit_transform(df["GpsProvider"])
    df["booking_prefix_enc"] = le.fit_transform(df["booking_prefix"])
    df["origin_city_enc"] = le.fit_transform(df["origin_city"])
    df["dest_city_enc"] = le.fit_transform(df["dest_city"])
    df["customerID_enc"] = le.fit_transform(df["customerID"])

    return df


def prepare_catboost_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """将 CatBoost / XGBoost 使用的类别特征转为 ``category`` 类型。"""
    df = df.copy()
    for col in XGB_CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).astype("category")
    return df


# ──────────────── 一键管道 ────────────────

def engineer_features(
    df: pd.DataFrame,
    run_clean: bool = False,
    years: list[int] | None = None,
    geo_radius: float = 3.0,
) -> pd.DataFrame:
    """完整特征工程管道。

    如果 ``run_clean=True``，先执行数据清洗再执行特征工程；
    否则假设 ``df`` 已经过清洗（含 ``Answer``、日期已解析、字段已填充）。

    Args:
        df: 原始或已清洗的 DataFrame。
        run_clean: 是否先运行清洗管道。
        years: 保留的年份。
        geo_radius: 地理填补搜索半径。

    Returns:
        含所有特征的 DataFrame（约 53 列）。
    """
    print("=" * 50)
    print("  特征工程管道")
    print("=" * 50)

    # 确保 Answer 列存在（原始数据用 ontime 创建，已清洗数据跳过）
    if "Answer" not in df.columns:
        df["Answer"] = df["ontime"].apply(lambda x: 0 if x == "G" else 1)
        print("[engineering] 从 ontime 创建 Answer 列")

    if run_clean:
        df = clean_data(df, years=years, geo_radius=geo_radius)
    else:
        print("[engineering] 跳过清洗，使用已清洗数据")

    df = add_time_features(df)
    df = add_min_kms_bin(df)
    df = add_planned_days_features(df)
    df = add_business_flags(df)
    df = encode_label_features(df)
    df = prepare_catboost_categoricals(df)

    print(f"[engineering] 特征工程完成, 形状: {df.shape}, "
          f"总列数: {len(df.columns)}")
    print(f"[engineering] 特征数量: {len(FEATURES_ENC)} (sklearn) / "
          f"{len(FEATURES_XGB)} (XGBoost/CatBoost)")
    return df
