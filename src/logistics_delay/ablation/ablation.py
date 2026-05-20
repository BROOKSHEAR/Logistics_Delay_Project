"""
特征消融实验脚本 (Feature Ablation)。

提供数据加载、留一法特征消融、学习曲线分析、结果保存功能。
只使用时序划分 (temporal split)，按 ``trip_start_date`` 排序后 80/20 切分。

用法:
    python -m src.logistics_delay.ablation.feature_ablation
"""
from __future__ import annotations

import os
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

from logistics_delay.utils.paths import DATA_RAW, SEED, TABLES_DIR
from logistics_delay.features.engineering import FEATURES_XGB, XGB_CAT_COLS
from logistics_delay.features.distance_fill_geo import DistanceFiller

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
#  1. 数据加载与特征工程
# ════════════════════════════════════════════════════════════════

def load_and_prepare_data() -> pd.DataFrame:
    """从预处理文件加载特征数据（跳过 raw → engineer_features 重复流程）。

    读取 ``data/processed/truck_delay_handled_file.xlsx``，
    并按 ``trip_start_date`` 时序排序后返回。

    Returns:
        排序后的完整 DataFrame，含 ``Answer`` 列和全部 61 列特征。
    """
    from logistics_delay.data.loader import load_processed
    df = load_processed()
    df_sorted = df.sort_values("trip_start_date").reset_index(drop=True)
    print(f"[data] 最终形状: {df_sorted.shape}")
    print(f"[data] 正样本比例: {df_sorted['Answer'].mean():.4f}")
    return df_sorted


# ════════════════════════════════════════════════════════════════
#  2. 特征消融 (Leave-One-Out)
# ════════════════════════════════════════════════════════════════

_DEFAULT_CB_PARAMS: dict = {
    "iterations": 200,
}


def _build_cb_model(
    spw: float,
    cat_features: list[str] | None,
    extra_params: dict | None = None,
) -> CatBoostClassifier:
    """构建 CatBoostClassifier，融合默认参数 + 调优参数 + 动态参数。"""
    params = dict(_DEFAULT_CB_PARAMS)
    if extra_params:
        params.update(extra_params)
    params.update(
        class_weights={0: 1.0, 1: spw},
        random_seed=SEED,
        verbose=0,
        cat_features=cat_features if cat_features else None,
    )
    return CatBoostClassifier(**params)


def run_feature_ablation(
    df_sorted: pd.DataFrame,
    feature_list: list[str] | None = None,
    catboost_params: dict | None = None,
) -> pd.DataFrame:
    """留一法特征消融 (Leave-One-Out)，使用 CatBoost。

    按 ``trip_start_date`` 时序 80/20 划分后，
    先训练全特征模型作为基准，再每次剔除一个特征重新训练，
    记录 AUC 和 F1 的变化。类别特征通过 ``cat_features`` 传入 CatBoost。

    Args:
        df_sorted: 经 ``load_and_prepare_data`` 处理并按时间排序的 DataFrame。
        feature_list: 参与消融的特征列表（默认 ``FEATURES_XGB``）。

    Returns:
        消融结果 DataFrame，列:
        - ``removed_feat``: 被剔除的特征名（首行为 ``（基准全特征）``）
        - ``auc``: 验证集 AUC
        - ``f1``: 验证集 F1
        - ``auc_drop``: ``全特征 AUC - 剔除后 AUC``（百分点）。
          正值表示去掉该特征后性能下降（特征重要），
          负值表示去掉后性能上升（特征有噪声）。
        - ``f1_drop``: ``全特征 F1 - 剔除后 F1``（百分点），符号含义同上。
    """
    if feature_list is None:
        feature_list = FEATURES_XGB

    print("\n" + "=" * 50)
    print("  特征消融实验 (CatBoost Leave-One-Out)")
    print("=" * 50)

    # ── 时序划分 80/20 ──
    split_idx = int(len(df_sorted) * 0.8)
    X_train = df_sorted.loc[: split_idx - 1, feature_list].reset_index(drop=True)
    X_test = df_sorted.loc[split_idx:, feature_list].reset_index(drop=True)
    y_train = df_sorted.loc[: split_idx - 1, "Answer"].reset_index(drop=True)
    y_test = df_sorted.loc[split_idx:, "Answer"].reset_index(drop=True)
    spw = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    print(f"[split] 训练: {len(y_train)} / 测试: {len(y_test)}, spw={spw:.4f}")

    # ── 全特征基准模型 ──
    cat_feats_full = [c for c in XGB_CAT_COLS if c in feature_list]
    model_full = _build_cb_model(spw, cat_feats_full, catboost_params)
    model_full.fit(X_train, y_train)
    y_pred_full = model_full.predict(X_test)
    y_prob_full = model_full.predict_proba(X_test)[:, 1]
    full_auc = roc_auc_score(y_test, y_prob_full)
    full_f1 = f1_score(y_test, y_pred_full)
    print(f"[full] AUC = {full_auc * 100:.2f}%  F1 = {full_f1:.4f}")

    # ── 留一法消融 ──
    results = []
    for feat in feature_list:
        subset = [f for f in feature_list if f != feat]
        cat_feats_sub = [c for c in cat_feats_full if c != feat]

        model = _build_cb_model(spw, cat_feats_sub, catboost_params)
        model.fit(X_train[subset], y_train)
        y_pred = model.predict(X_test[subset])
        y_prob = model.predict_proba(X_test[subset])[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        auc_drop_val = round((full_auc - auc) * 100, 4)
        f1_drop_val = round((full_f1 - f1) * 100, 4)
        results.append({
            "removed_feat": feat,
            "auc": round(auc, 6),
            "f1": round(f1, 6),
            "auc_drop": auc_drop_val,
            "f1_drop": f1_drop_val,
        })
        print(f"  去掉 {feat:<30s}  AUC = {auc * 100:.2f}%  "
              f"(drop={auc_drop_val:+.2f}pp)  "
              f"F1 = {f1:.4f}  (drop={f1_drop_val:+.4f}pp)")

    # 插入全特征基准行
    full_row = pd.DataFrame([{
        "removed_feat": "（全特征基准）",
        "auc": round(full_auc, 6),
        "f1": round(full_f1, 6),
        "auc_drop": 0.0,
        "f1_drop": 0.0,
    }])
    result_df = pd.concat([full_row, pd.DataFrame(results)], ignore_index=True)
    return result_df


# ════════════════════════════════════════════════════════════════
#  3. 地理消融
# ════════════════════════════════════════════════════════════════

def run_geo_ablation(
    df_sorted: pd.DataFrame,
    catboost_params: dict | None = None,
) -> pd.DataFrame:
    """地理消融：对比三种距离填充方案对 CatBoost 性能的影响。

    三种方案使用完全相同的其他特征和模型超参数，仅对
    ``TRANSPORTATION_DISTANCE_IN_KM`` 的缺失值采用不同填充策略：
      - **地理填充**: 使用 ``DistanceFiller`` 地理邻近性填补 + 中位数兜底。
      - **中位数填充**: 所有缺失值直接使用中位数填充。
      - **均值填充**: 所有缺失值直接使用均值填充。

    Args:
        df_sorted: 经 ``load_and_prepare_data`` 处理并按时间排序的 DataFrame。
           ``_dist_original`` 列由 ``cleaner.fill_distance_geo`` 在预处理时保存，
           记录地理填充前的原始距离值（含 NaN），标识哪些行的距离是缺失的。

    Returns:
        地理消融结果 DataFrame，列:
        - ``strategy``: 填充方案名称
        - ``auc``: 验证集 AUC
        - ``f1``: 验证集 F1
    """
    print("\n" + "=" * 50)
    print("  地理消融实验")
    print("=" * 50)

    # ── 提取原始距离信息 ──
    missing_mask = df_sorted["_dist_original"].isna()
    orig_dist = df_sorted["_dist_original"].values
    non_missing = orig_dist[~missing_mask.values]
    median_val = np.median(non_missing)
    mean_val = np.mean(non_missing)
    print(f"  缺失距离: {missing_mask.sum()} / {len(orig_dist)}")
    print(f"  中位数: {median_val:.2f}, 均值: {mean_val:.2f}")

    # ── 三种策略 ──
    strategies: dict[str, float | None] = {
        "geo": None,
        "median": median_val,
        "mean": mean_val,
    }
    strategy_labels = {
        "geo": "Geo",
        "median": "Median",
        "mean": "Mean",
    }

    feature_list = FEATURES_XGB
    cat_feats = [c for c in XGB_CAT_COLS if c in feature_list]
    split_idx = int(len(df_sorted) * 0.8)

    results = []
    for key, fill_value in strategies.items():
        df_copy = df_sorted.copy()

        if fill_value is None:
            # 地理填充：使用 df_sorted 已有的地理 + 中位数兜底值（不变）
            pass
        else:
            # 中位数 / 均值填充
            dist_col = orig_dist.copy()
            dist_col[missing_mask.values] = fill_value
            df_copy["TRANSPORTATION_DISTANCE_IN_KM"] = dist_col

        X_tr = df_copy.loc[: split_idx - 1, feature_list].reset_index(drop=True)
        X_te = df_copy.loc[split_idx:, feature_list].reset_index(drop=True)
        y_tr = df_copy.loc[: split_idx - 1, "Answer"].reset_index(drop=True)
        y_te = df_copy.loc[split_idx:, "Answer"].reset_index(drop=True)
        spw = len(y_tr[y_tr == 0]) / max(len(y_tr[y_tr == 1]), 1)

        model = _build_cb_model(spw, cat_feats, catboost_params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
        f1 = f1_score(y_te, y_pred)

        results.append({
            "strategy": strategy_labels[key],
            "auc": round(auc, 6),
            "f1": round(f1, 6),
        })
        print(f"  {strategy_labels[key]:<20s}  AUC={auc * 100:.2f}%  F1={f1:.4f}")

    return pd.DataFrame(results)


def save_geo_results(
    geo_df: pd.DataFrame,
    save_dir: str | os.PathLike | None = None,
) -> None:
    """将地理消融结果保存为 CSV。

    Args:
        geo_df: ``run_geo_ablation`` 返回的 DataFrame。
        save_dir: 保存目录（默认 ``TABLES_DIR``）。
    """
    if save_dir is None:
        save_dir = TABLES_DIR
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "geo_ablation_results.csv")
    geo_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] 地理消融结果 → {path}")


# ════════════════════════════════════════════════════════════════
#  4. 累积特征学习曲线
# ════════════════════════════════════════════════════════════════

def run_learning_curves(
    df_sorted: pd.DataFrame,
    feature_list: list[str] | None = None,
    ablation_df: pd.DataFrame | None = None,
    catboost_params: dict | None = None,
) -> pd.DataFrame:
    """累积特征学习曲线。

    先通过 ``run_feature_ablation`` 按 ``auc_drop`` 从大到小得到特征重要性排序，
    然后从最重要的特征开始依次累积加入，每加入一个特征就用 CatBoost 时序划分
    训练一次并记录 AUC，直到加入全部特征。

    Args:
        df_sorted: 经 ``load_and_prepare_data`` 处理并按时间排序的 DataFrame。
        feature_list: 特征列表（默认 ``FEATURES_XGB``）。

    Returns:
        学习曲线结果 DataFrame，列:
        - ``feature_added``: 本轮新加入的特征名
        - ``n_features``: 当前已加入特征数量
        - ``auc``: 验证集 AUC
    """
    if feature_list is None:
        feature_list = FEATURES_XGB

    print("\n" + "=" * 50)
    print("  累积特征学习曲线")
    print("=" * 50)

    # ── 1. 通过消融结果获取特征重要性排序 ──
    if ablation_df is None:
        ablation_df = run_feature_ablation(df_sorted, feature_list)
    imp_df = ablation_df[ablation_df["removed_feat"] != "（全特征基准）"].copy()
    imp_df = imp_df.sort_values("auc_drop", ascending=False)
    feature_order = imp_df["removed_feat"].tolist()
    print(f"\n  特征加入顺序（按 auc_drop 降序）:")
    for i, f in enumerate(feature_order, 1):
        print(f"    {i:>2d}. {f}")

    # ── 2. 时序划分 ──
    split_idx = int(len(df_sorted) * 0.8)
    y_train = df_sorted.loc[: split_idx - 1, "Answer"].reset_index(drop=True)
    y_test = df_sorted.loc[split_idx:, "Answer"].reset_index(drop=True)
    cat_feats_all = [c for c in XGB_CAT_COLS if c in feature_list]

    print(f"\n[split] 训练: {split_idx} / 测试: {len(df_sorted) - split_idx}")

    # ── 3. 累积加入特征 ──
    records = []
    cumulative: list[str] = []
    for feat in feature_order:
        cumulative.append(feat)
        X_tr = df_sorted.loc[: split_idx - 1, cumulative].reset_index(drop=True)
        X_te = df_sorted.loc[split_idx:, cumulative].reset_index(drop=True)
        cat_feats = [c for c in cat_feats_all if c in cumulative]

        spw = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
        model = _build_cb_model(spw, cat_feats, catboost_params)
        model.fit(X_tr, y_train)
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        records.append({
            "feature_added": feat,
            "n_features": len(cumulative),
            "auc": round(auc, 6),
        })
        print(f"  +{feat:<30s}  n={len(cumulative):>2d}  AUC={auc * 100:.2f}%")

    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════
#  5. 结果保存
# ════════════════════════════════════════════════════════════════

def save_results(
    ablation_df: pd.DataFrame,
    lc_df: pd.DataFrame,
    save_dir: str | os.PathLike | None = None,
) -> None:
    """将特征消融和学习曲线结果保存为 CSV。

    Args:
        ablation_df: ``run_feature_ablation`` 返回的 DataFrame。
            ``auc_drop`` = 全特征 AUC - 剔除后 AUC（正 = 性能下降，负 = 性能提升）。
        lc_df: ``run_learning_curves`` 返回的 DataFrame。
            含 ``feature_added``、``n_features``、``auc`` 三列。
        save_dir: 保存目录（默认 ``TABLES_DIR``）。
    """
    if save_dir is None:
        save_dir = TABLES_DIR

    os.makedirs(save_dir, exist_ok=True)

    ablation_path = os.path.join(save_dir, "feature_ablation_results.csv")
    lc_path = os.path.join(save_dir, "learning_curves_results.csv")

    ablation_df.to_csv(ablation_path, index=False, encoding="utf-8-sig")
    lc_df.to_csv(lc_path, index=False, encoding="utf-8-sig")

    print(f"\n[OK] 特征消融结果 → {ablation_path}")
    print(f"[OK] 学习曲线结果 → {lc_path}")


# ════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  消融实验脚本 — 主入口")
    print("=" * 60)

    df_sorted = load_and_prepare_data()

    ablation_results = run_feature_ablation(df_sorted)
    lc_results = run_learning_curves(df_sorted, ablation_df=ablation_results)
    save_results(ablation_results, lc_results)
    print(df_sorted.columns.tolist())
    geo_results = run_geo_ablation(df_sorted)
    save_geo_results(geo_results)

    print("\n" + "=" * 60)
    print("  全部完成")
    print("=" * 60)
