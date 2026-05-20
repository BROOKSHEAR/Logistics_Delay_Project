"""
切割点敏感性分析：检验 CatBoost 在单次时序切分下的 test_auc 是否稳定。

遍历多个训练集占比 (70%, 75%, 80%, 85%)，
每个比例仅训练一次 CatBoost（用最佳超参数），
观察 test_auc 对切割点的敏感程度。

用法:
    python -m src.logistics_delay.ablation.split_sensitivity

输出:
    - outputs/tables/split_sensitivity.csv
    - 控制台打印对比表格
"""
from __future__ import annotations

import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from catboost import CatBoostClassifier

from logistics_delay.ablation.ablation import load_and_prepare_data
from logistics_delay.features.engineering import FEATURES_XGB, XGB_CAT_COLS
from logistics_delay.utils.paths import SEED, TABLES_DIR

warnings.filterwarnings("ignore")

# 最佳 CatBoost 超参数（来自 04_tuning.ipynb）
BEST_CB_PARAMS: dict = {
    "learning_rate": 0.05,
    "depth": 8,
    "iterations": 100,
    "l2_leaf_reg": 5,
    "border_count": 64,
    "bagging_temperature": 1,
    "random_strength": 0,
}


def run_sensitivity(
    df_sorted: pd.DataFrame,
    split_ratios: list[float] | None = None,
) -> pd.DataFrame:
    """对每个切割比例训练 CatBoost 并记录 test_auc。

    Args:
        df_sorted: 按 trip_start_date 排序后的完整 DataFrame。
        split_ratios: 训练集占比列表，默认 [0.70, 0.75, 0.80, 0.85]。

    Returns:
        DataFrame，每行一个切割比例，含:
        - train_ratio: 训练集占比
        - test_ratio: 测试集占比
        - train_range: 训练集日期范围
        - test_range: 测试集日期范围
        - train_size / test_size
        - pos_rate_train / pos_rate_test: 延误率
        - spw: scale_pos_weight
        - test_auc / test_f1
    """
    if split_ratios is None:
        split_ratios = [0.70, 0.75, 0.80, 0.85]

    feature_list = FEATURES_XGB
    cat_feats = [c for c in XGB_CAT_COLS if c in feature_list]

    records: list[dict] = []

    for ratio in split_ratios:
        split_idx = int(len(df_sorted) * ratio)

        X_train = df_sorted.loc[: split_idx - 1, feature_list].reset_index(drop=True)
        X_test = df_sorted.loc[split_idx:, feature_list].reset_index(drop=True)
        y_train = df_sorted.loc[: split_idx - 1, "Answer"].reset_index(drop=True)
        y_test = df_sorted.loc[split_idx:, "Answer"].reset_index(drop=True)

        spw = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)

        start_date = df_sorted["trip_start_date"]
        train_range = (
            f"{start_date.iloc[0].date()} ~ {start_date.iloc[split_idx - 1].date()}"
        )
        test_range = (
            f"{start_date.iloc[split_idx].date()} ~ {start_date.iloc[-1].date()}"
        )

        model = CatBoostClassifier(
            **BEST_CB_PARAMS,
            class_weights={0: 1.0, 1: spw},
            random_seed=SEED,
            verbose=0,
            cat_features=cat_feats,
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        records.append({
            "train_ratio": ratio,
            "test_ratio": round(1 - ratio, 2),
            "train_range": train_range,
            "test_range": test_range,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "pos_rate_train": round(float(y_train.mean()), 4),
            "pos_rate_test": round(float(y_test.mean()), 4),
            "spw": round(spw, 2),
            "test_auc": round(roc_auc_score(y_test, y_prob), 4),
            "test_f1": round(f1_score(y_test, y_pred), 4),
        })

    return pd.DataFrame(records)


def print_results(df: pd.DataFrame) -> None:
    """格式化打印敏感性分析结果。"""
    print("=" * 64)
    print("  CatBoost 切割点敏感性分析")
    print("=" * 64)
    print(
        f"  {'训练比':>6s}  {'测试比':>6s}  {'训练延误率':<10s}"
        f"  {'测试延误率':<10s}  {'SPW':>6s}  {'AUC':>7s}  {'F1':>7s}"
    )
    print("  " + "-" * 58)
    for _, row in df.iterrows():
        print(
            f"  {row['train_ratio']:>5.0%}  {row['test_ratio']:>5.0%}  "
            f"{row['pos_rate_train']:>7.2%}   {row['pos_rate_test']:>7.2%}   "
            f"{row['spw']:>6.2f}  {row['test_auc']:>7.4f}  {row['test_f1']:>7.4f}"
        )
    print("  " + "-" * 58)

    aucs = df["test_auc"].values
    print(f"  AUC 均值: {aucs.mean():.4f}  "
          f"标准差: {aucs.std():.4f}  "
          f"极差: {aucs.max() - aucs.min():.4f}")
    print(f"  解读: ", end="")
    if aucs.max() - aucs.min() < 0.01:
        print("[OK] AUC 波动 < 1pp，切割点影响不大，单次切分可用。")
    elif aucs.max() - aucs.min() < 0.02:
        print("[WARN] AUC 波动 1~2pp，存在一定敏感性，建议改用 TimeSeriesSplit。")
    else:
        print("[ALERT] AUC 波动 > 2pp，切割点高度敏感，必须改用 TimeSeriesSplit。")

    print("\n  各切割点日期范围:")
    for _, row in df.iterrows():
        print(f"    {row['train_ratio']:.0%}/{-row['test_ratio']:.0%}  "
              f"训练: {row['train_range']}")
        print(f"    {'':>6s}  测试: {row['test_range']}")


def save_results(df: pd.DataFrame) -> None:
    """保存为 CSV。"""
    import os
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = TABLES_DIR / "split_sensitivity.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 敏感性分析结果 → {path}")


if __name__ == "__main__":
    print("加载数据...")
    df_sorted = load_and_prepare_data()

    result_df = run_sensitivity(df_sorted)
    print_results(result_df)
    save_results(result_df)
