"""
Feature engineering module.

Provides independent temporal feature, binning, encoding functions, and ``engineer_features`` pipeline.

Usage:
    from logistics_delay.features.engineering import engineer_features, get_feature_lists
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from logistics_delay.data.cleaner import clean_data, fill_distance_geo
from logistics_delay.utils.paths import DATA_PROCESSED

# ──────────────── Feature list constants (global) ────────────────

XGB_CAT_COLS = [
    "vehicleType", "OriginLocation_Code", "DestinationLocation_Code",
    "GpsProvider", "booking_prefix", "origin_city", "dest_city", "customerID",
    "Minimum_kms_to_be_covered_in_a_day",
]

FEATURES_ENC = [
    "TRANSPORTATION_DISTANCE_IN_KM", "vehicleType_enc", "is_market",
    "start_weekday", "start_month",
    "OriginLocation_Code_enc", "DestinationLocation_Code_enc",
    "Minimum_kms_to_be_covered_in_a_day_enc", "planned_days_enc", "booking_prefix_enc",
    "origin_city_enc", "dest_city_enc", "customerID_enc",
    "supplier_is_large", "GpsProvider_enc",
]

FEATURES_XGB = [
    "TRANSPORTATION_DISTANCE_IN_KM", "is_market",
    "start_weekday", "start_month",
     "planned_days_enc",
    "supplier_is_large",
] + XGB_CAT_COLS


def get_feature_lists():
    """Return the three feature list constants.

    Returns:
        (features_enc, features_xgb, xgb_cat_cols) tuple.
    """
    return FEATURES_ENC, FEATURES_XGB, XGB_CAT_COLS


# ──────────────── Low-level independent functions ────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``start_weekday`` (0=Mon) and ``start_month`` (1=Jan)."""
    df = df.copy()
    df["start_weekday"] = df["trip_start_date"].dt.dayofweek
    df["start_month"] = df["trip_start_date"].dt.month
    return df


def add_planned_days_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ``planned_days`` and its binned encoding ``planned_days_enc``.

    Compute planned days from ``Planned_ETA`` and ``BookingID_Date``,
    remove negatives and >180 day outliers, fill with median.
    Binning: [-1, 1] → 0, (1, 3] → 1, (3, 7] → 2, (7, 999] → 3.
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
    """Add business indicator features ``is_market`` and ``supplier_is_large``."""
    df = df.copy()
    df["is_market"] = (df["Market/Regular"] == "Market").astype(int)
    has_alpha = df["supplierID"].str.contains("[A-Za-z]", regex=True, na=False)
    df["supplier_is_large"] = has_alpha.astype(int)
    return df


def encode_label_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply LabelEncoder to all categorical features, generating ``*_enc`` columns."""
    df = df.copy()
    le = LabelEncoder()

    for col in ["vehicleType", "OriginLocation_Code", "DestinationLocation_Code"]:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    df["GpsProvider_enc"] = le.fit_transform(df["GpsProvider"])
    df["booking_prefix_enc"] = le.fit_transform(df["booking_prefix"])
    df["origin_city_enc"] = le.fit_transform(df["origin_city"])
    df["dest_city_enc"] = le.fit_transform(df["dest_city"])
    df["customerID_enc"] = le.fit_transform(df["customerID"])
    df["Minimum_kms_to_be_covered_in_a_day_enc"] = le.fit_transform(
        df["Minimum_kms_to_be_covered_in_a_day"].astype(str)
    )

    return df


def prepare_catboost_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical features used by CatBoost / XGBoost to ``category`` dtype."""
    df = df.copy()
    for col in XGB_CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).astype("category")
    return df


# ──────────────── Pipeline ────────────────

def engineer_features(
    df: pd.DataFrame,
    run_clean: bool = False,
    save_processed: bool = False,
    years: list[int] | None = None,
    geo_radius: float = 3.0,
) -> pd.DataFrame:
    """Full feature engineering pipeline.

    If ``run_clean=True``, run cleaning first, then feature engineering;
    otherwise assume ``df`` is already cleaned (has ``Answer``, parsed dates, filled fields).

    Args:
        df: Raw or cleaned DataFrame.
        run_clean: Whether to run cleaning first.
        years: Years to retain.
        geo_radius: Geo-fill search radius.

    Returns:
        DataFrame with all features (~53 columns).
    """
    print("=" * 50)
    print("  Feature engineering pipeline")
    print("=" * 50)

    # Ensure Answer column exists (create from ontime for raw data, skip if already cleaned)
    if "Answer" not in df.columns:
        df["Answer"] = df["ontime"].apply(lambda x: 0 if x == "G" else 1)
        print("[engineering] Creating Answer column from ontime")

    if run_clean:
        df = clean_data(df, years=years, geo_radius=geo_radius)
    else:
        print("[engineering] Skipping cleaning, using pre-cleaned data")

    df = add_time_features(df)
    df = add_planned_days_features(df)
    df = add_business_flags(df)
    df = encode_label_features(df)
    df = prepare_catboost_categoricals(df)

    print(f"[engineering] Feature engineering done, shape: {df.shape}, "
          f"total columns: {len(df.columns)}")
    print(f"[engineering] Features: {len(FEATURES_ENC)} (sklearn) / "
          f"{len(FEATURES_XGB)} (XGBoost/CatBoost)")

    if save_processed:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        path = DATA_PROCESSED / "truck_delay_handled_file.xlsx"
        df.to_excel(path, index=False)
        print(f"[engineering] Saved → {path} ({df.shape[0]} rows × {len(df.columns)} cols)")

    return df
