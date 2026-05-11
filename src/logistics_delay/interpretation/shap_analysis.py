"""
SHAP 可解释性分析模块。

提供一键式计算 SHAP 值、输出排名、与消融实验对比的相关性分析。
实际的绘图（蜂群图、条形图）保留在 notebook 中执行，以便交互调整。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import shap


def compute_shap_importance(model, X_test: pd.DataFrame,
                            feature_names: list[str] | None = None,
                            model_type: str = "tree") -> pd.DataFrame:
    """计算 SHAP 值并返回特征重要性排名 DataFrame。

    Args:
        model: 已训练的模型（支持 tree / linear / kernel explainer）。
        X_test: 测试特征矩阵。
        feature_names: 特征名列表，默认使用 ``X_test.columns``。
        model_type: ``"tree"``, ``"linear"`` 或 ``"kernel"``。

    Returns:
        DataFrame 按 ``mean_abs_shap`` 降序排列。
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X_test)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_test)

    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        # 多分类场景取正类（索引 1）
        shap_values = shap_values[1]

    feat_names = feature_names or list(X_test.columns)
    imp_df = pd.DataFrame({
        "feature": feat_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(f"\n[shap] SHAP 重要性排名:")
    for i, (_, row) in enumerate(imp_df.iterrows()):
        print(f"  {i + 1}. {row['feature']:35s}  {row['mean_abs_shap']:.6f}")

    return imp_df, shap_values, explainer


def compare_with_ablation(shap_df: pd.DataFrame,
                          ablation_df: pd.DataFrame,
                          split_name: str = "random") -> float:
    """计算 SHAP 重要性排名与特征消融 AUC 下降量的 Pearson 相关系数。

    高相关系数意味着 SHAP 分析与消融实验结论一致，
    增强了特征重要性排序的可信度。

    Args:
        shap_df: ``compute_shap_importance`` 返回的 DataFrame。
        ablation_df: 含 ``removed_feat`` 和 ``auc_drop`` 列的消融结果。
        split_name: 划分方式（仅用于打印）。

    Returns:
        Pearson 相关系数。
    """
    merged = pd.merge(
        shap_df,
        ablation_df[["removed_feat", "auc_drop"]],
        left_on="feature",
        right_on="removed_feat",
        how="left",
    )
    pearson_r = merged["mean_abs_shap"].corr(merged["auc_drop"].abs())

    print(f"\n[shap] SHAP vs 消融 Pearson 相关系数 ({split_name}): {pearson_r:.4f}")
    return pearson_r
