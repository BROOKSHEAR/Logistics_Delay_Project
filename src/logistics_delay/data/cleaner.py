"""
数据清洗模块。

提供独立的清洗函数（可单独使用），以及一键式 ``clean_data`` 完整管道。
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from logistics_delay.features.distance_fill_geo import DistanceFiller


# ──────────────── 底层独立函数（可单独调用） ────────────────

def remove_conflict_rows(df: pd.DataFrame) -> pd.DataFrame:
    """删除 ``ontime`` 与 ``delay`` 矛盾的行（ontime=G 但 delay=R）。

    这类数据占比约 0.35%，属于逻辑错误。
    """
    n_before = len(df)
    has_delay = df["delay"] == "R"
    conflict = has_delay & (df["ontime"] == "G")
    df_clean = df[~conflict].reset_index(drop=True)
    n_removed = n_before - len(df_clean)
    print(f"[cleaner] 删除 {n_removed} 行冲突数据 ({n_removed / n_before * 100:.2f}%)")
    print(f"[cleaner] 清洗冲突后: {len(df_clean)} 行")
    return df_clean


def parse_and_filter_dates(df: pd.DataFrame,
                           years: list[int] | None = None) -> pd.DataFrame:
    """解析 ``trip_start_date`` 并过滤指定年份。

    Args:
        df: 输入 DataFrame（需含 ``trip_start_date`` 列）。
        years: 保留的年份列表，默认 ``[2019, 2020]``。

    Returns:
        过滤后的 DataFrame。
    """
    if years is None:
        years = [2019, 2020]

    df = df.copy()
    df["trip_start_date"] = pd.to_datetime(df["trip_start_date"], errors="coerce")
    df["year"] = df["trip_start_date"].dt.year

    print(f"[cleaner] 原始日期范围: {df['trip_start_date'].min().date()} ~ "
          f"{df['trip_start_date'].max().date()}")

    n_before = len(df)
    df = df[df["year"].isin(years)].reset_index(drop=True)
    n_removed = n_before - len(df)
    print(f"[cleaner] 过滤掉 {n_removed} 行非 {years} 数据")
    print(f"[cleaner] 保留 {len(df)} 行")

    return df


def fill_distance_geo(df: pd.DataFrame,
                      max_search_radius: float = 3.0,
                      verbose: bool = False) -> pd.DataFrame:
    """使用地理邻近性填补运输距离缺失值。

    Args:
        df: 输入 DataFrame。
        max_search_radius: 最大搜索半径（经纬度）。
        verbose: 是否打印每条填补记录。

    Returns:
        距离已填补的 DataFrame。
    """
    df = df.copy()
    missing_mask = df["TRANSPORTATION_DISTANCE_IN_KM"].isna()
    n_missing = int(missing_mask.sum())
    print(f"[cleaner] 运输距离缺失: {n_missing} 条")

    if n_missing == 0:
        return df

    filler = DistanceFiller(df)
    df_filled = filler.fill_missing_distances(
        max_search_radius=max_search_radius, verbose=verbose
    )
    df["TRANSPORTATION_DISTANCE_IN_KM"] = df_filled[
        "TRANSPORTATION_DISTANCE_IN_KM"
    ]

    n_still = df["TRANSPORTATION_DISTANCE_IN_KM"].isna().sum()
    print(f"[cleaner] 地理填补完成，共填补 {n_missing - n_still} 条")

    if n_still > 0:
        median_val = df["TRANSPORTATION_DISTANCE_IN_KM"].median()
        df["TRANSPORTATION_DISTANCE_IN_KM"] = (
            df["TRANSPORTATION_DISTANCE_IN_KM"].fillna(median_val)
        )
        print(f"[cleaner] 使用中位数 {median_val:.1f} km 填补剩余 {n_still} 条")

    return df


def fill_basic_fields(df: pd.DataFrame) -> pd.DataFrame:
    """填充其他基本字段的缺失值并提取派生字段。

    包括: Minimum_kms, vehicleType, GpsProvider, booking_prefix,
    origin_city, dest_city, Code/Customer/Supplier 字段。
    """
    df = df.copy()

    # Minimum_kms
    median_kms = df["Minimum_kms_to_be_covered_in_a_day"].median()
    df["Minimum_kms_to_be_covered_in_a_day"] = (
        df["Minimum_kms_to_be_covered_in_a_day"].fillna(median_kms)
    )
    print(f"[cleaner] Minimum_kms: 用中位数 {median_kms:.1f} 填充")

    # vehicleType
    df["vehicleType"] = df["vehicleType"].fillna("Unknown")

    # BookingID_Date 解析
    df["BookingID_Date"] = pd.to_datetime(
        df["BookingID_Date"], unit="D", origin="1899-12-30"
    )

    # GpsProvider
    df["GpsProvider"] = (
        df["GpsProvider"].fillna("UNKNOWN").str.strip().str.upper()
    )

    # booking_prefix
    df["booking_prefix"] = (
        df["BookingID"].astype(str).str.extract(r"^([A-Za-z]+)")[0].fillna("UNKNOWN")
    )

    # 城市字段
    df["origin_city"] = (
        df["Origin_Location"]
        .astype(str)
        .str.split(",")
        .str[1]
        .str.strip()
        .str.upper()
        .fillna("UNKNOWN")
    )
    df["dest_city"] = (
        df["Destination_Location"]
        .astype(str)
        .str.split(",")
        .str[1]
        .str.strip()
        .str.upper()
        .fillna("UNKNOWN")
    )

    # 其他 Code 字段
    df["OriginLocation_Code"] = df["OriginLocation_Code"].fillna("UNKNOWN")
    df["DestinationLocation_Code"] = df["DestinationLocation_Code"].fillna("UNKNOWN")
    df["customerID"] = df["customerID"].fillna("UNKNOWN").astype(str)
    df["supplierID"] = df["supplierID"].fillna("UNKNOWN").astype(str)

    print(f"[cleaner] 基本字段填充完成")
    return df


# ──────────────── 一键管道 ────────────────

def clean_data(df: pd.DataFrame,
               years: list[int] | None = None,
               geo_radius: float = 3.0) -> pd.DataFrame:
    """完整数据清洗管道：冲突删除 → 日期过滤 → 距离填补 → 字段填充。

    Args:
        df: 原始 DataFrame（含 ``Answer`` 列）。
        years: 保留的年份。
        geo_radius: 地理填补搜索半径。

    Returns:
        清洗完成的 DataFrame。
    """
    print("=" * 50)
    print("  数据清洗管道")
    print("=" * 50)

    df = remove_conflict_rows(df)
    df = parse_and_filter_dates(df, years=years)
    df = fill_distance_geo(df, max_search_radius=geo_radius)
    df = fill_basic_fields(df)

    # 最终检查
    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]
    if len(remaining) > 0:
        print(f"[cleaner] ⚠ 仍有缺失值的字段: {remaining.to_dict()}")
    else:
        print("[cleaner] ✅ 所有缺失值已处理")

    print(f"[cleaner] 清洗完成, 形状: {df.shape}, "
          f"延误率: {df['Answer'].mean() * 100:.2f}%")
    return df
