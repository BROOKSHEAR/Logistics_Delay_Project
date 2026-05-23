"""
Data cleaning module.

Provides independent cleaning functions (usable standalone) and ``clean_data`` pipeline.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from logistics_delay.features.distance_fill_geo import DistanceFiller


# ──────────────── Low-level independent functions (callable standalone) ────────────────

def remove_conflict_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where ``ontime`` and ``delay`` contradict (ontime=G but delay=R).

    These account for ~0.35% of data and are logical errors.
    """
    n_before = len(df)
    has_delay = df["delay"] == "R"
    conflict = has_delay & (df["ontime"] == "G")
    df_clean = df[~conflict].reset_index(drop=True)
    n_removed = n_before - len(df_clean)
    print(f"[cleaner] Removed {n_removed} conflicting rows ({n_removed / n_before * 100:.2f}%)")
    print(f"[cleaner] After conflict cleaning: {len(df_clean)} rows")
    return df_clean


def parse_and_filter_dates(df: pd.DataFrame,
                           years: list[int] | None = None) -> pd.DataFrame:
    """Parse ``trip_start_date`` and filter to specified years.

    Args:
        df: Input DataFrame (must have ``trip_start_date`` column).
        years: Years to keep, default ``[2019, 2020]``.

    Returns:
        Filtered DataFrame.
    """
    if years is None:
        years = [2019, 2020]

    df = df.copy()
    df["trip_start_date"] = pd.to_datetime(df["trip_start_date"], errors="coerce")
    df["year"] = df["trip_start_date"].dt.year

    print(f"[cleaner] Original date range: {df['trip_start_date'].min().date()} ~ "
          f"{df['trip_start_date'].max().date()}")

    n_before = len(df)
    df = df[df["year"].isin(years)].reset_index(drop=True)
    n_removed = n_before - len(df)
    print(f"[cleaner] Filtered out {n_removed} rows not in {years}")
    print(f"[cleaner] Retained {len(df)} rows")

    return df


def fill_distance_geo(df: pd.DataFrame,
                      max_search_radius: float = 3.0,
                      verbose: bool = False) -> pd.DataFrame:
    """Fill missing transportation distance using geographic proximity.

    Args:
        df: Input DataFrame.
        max_search_radius: Maximum search radius (lat/lon degrees).
        verbose: Whether to print each fill record.

    Returns:
        DataFrame with filled distances.
    """
    df = df.copy()
    # Save original distance (with NaN) for geo ablation to compare strategies
    df["_dist_original"] = df["TRANSPORTATION_DISTANCE_IN_KM"].copy()
    missing_mask = df["TRANSPORTATION_DISTANCE_IN_KM"].isna()
    n_missing = int(missing_mask.sum())
    print(f"[cleaner] Distance missing: {n_missing} records")

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
    print(f"[cleaner] Geo imputation done, filled {n_missing - n_still} records")

    if n_still > 0:
        median_val = df["TRANSPORTATION_DISTANCE_IN_KM"].median()
        df["TRANSPORTATION_DISTANCE_IN_KM"] = (
            df["TRANSPORTATION_DISTANCE_IN_KM"].fillna(median_val)
        )
        print(f"[cleaner] Median {median_val:.1f} km used for remaining {n_still} records")

    return df


def fill_basic_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in basic fields and extract derived fields.

    Includes: Minimum_kms, vehicleType, GpsProvider, booking_prefix,
    origin_city, dest_city, Code/Customer/Supplier fields.
    """
    df = df.copy()

    # Minimum_kms (categorical)
    df["Minimum_kms_to_be_covered_in_a_day"] = (
        df["Minimum_kms_to_be_covered_in_a_day"]
        .fillna("UNKNOWN")
        .astype(str)
    )
    print("[cleaner] Minimum_kms_to_be_covered_in_a_day: categorical, missing filled as UNKNOWN")

    # vehicleType
    df["vehicleType"] = df["vehicleType"].fillna("Unknown")

    # Parse BookingID_Date
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

    # City fields
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

    # Other Code fields
    df["OriginLocation_Code"] = df["OriginLocation_Code"].fillna("UNKNOWN")
    df["DestinationLocation_Code"] = df["DestinationLocation_Code"].fillna("UNKNOWN")
    df["customerID"] = df["customerID"].fillna("UNKNOWN").astype(str)
    df["supplierID"] = df["supplierID"].fillna("UNKNOWN").astype(str)

    print(f"[cleaner] Basic field filling complete")
    return df


# ──────────────── Pipeline ────────────────

def clean_data(df: pd.DataFrame,
               years: list[int] | None = None,
               geo_radius: float = 3.0) -> pd.DataFrame:
    """Full data cleaning pipeline: conflict removal → date filtering → distance imputation → field filling.

    Args:
        df: Raw DataFrame (must have ``Answer`` column).
        years: Years to retain.
        geo_radius: Geo imputation search radius.

    Returns:
        Cleaned DataFrame.
    """
    print("=" * 50)
    print("  Data cleaning pipeline")
    print("=" * 50)

    df = remove_conflict_rows(df)
    df = parse_and_filter_dates(df, years=years)
    df = fill_distance_geo(df, max_search_radius=geo_radius)
    df = fill_basic_fields(df)

    # Final check
    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]
    if len(remaining) > 0:
        print(f"[cleaner] Fields still containing NaNs: {remaining.to_dict()}")
    else:
        print("[cleaner] All missing values handled")

    print(f"[cleaner] Cleaning complete, shape: {df.shape}, "
          f"delay rate: {df['Answer'].mean() * 100:.2f}%")
    return df
