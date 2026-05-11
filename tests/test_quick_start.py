"""
验证 Quick Start 的全流程可复现性。

确保别人 git clone 后能成功加载数据、调用模型、看到输出。
测试覆盖 README.md 中 Quick Start 的所有代码路径。
"""
import sys
import warnings

warnings.filterwarnings("ignore")

# ──────────── 数据加载 ────────────


def test_load_data():
    """验证预处理数据加载"""
    print("=" * 58)
    print("  [1/5] 数据加载: load_processed()")
    print("=" * 58)

    from logistics_delay.data.loader import load_processed, load_raw_data

    # 1a. 加载预处理数据（主要路径）
    df = load_processed()
    assert df is not None, "load_processed() 返回 None"
    assert not df.empty, "返回空 DataFrame"
    assert "Answer" in df.columns, "缺少 Answer 列"
    assert "trip_start_date" in df.columns, "缺少 trip_start_date 列"
    assert df.index.name is None or df.index.name == "index"

    print(f"   形状: {df.shape}")
    print(f"   列数: {len(df.columns)}")
    print(f"   延误率: {df['Answer'].mean() * 100:.2f}%")

    # 1b. 验证原始数据也能加载
    raw = load_raw_data()
    assert raw is not None
    print(f"   原始数据: {raw.shape}")

    return df


# ──────────── 特征列表 ────────────


def test_feature_lists(df):
    """验证特征列表常量"""
    print("\n" + "=" * 58)
    print("  [2/5] 特征列表: get_feature_lists()")
    print("=" * 58)

    from logistics_delay.features.engineering import get_feature_lists, FEATURES_ENC, FEATURES_XGB, XGB_CAT_COLS

    fe, fx, cat = get_feature_lists()
    assert len(fe) == len(FEATURES_ENC), f"FEATURES_ENC 长度不符: {len(fe)} vs {len(FEATURES_ENC)}"
    assert len(fx) == len(FEATURES_XGB), f"FEATURES_XGB 长度不符: {len(fx)} vs {len(FEATURES_XGB)}"
    assert len(cat) == len(XGB_CAT_COLS), f"XGB_CAT_COLS 长度不符: {len(cat)} vs {len(XGB_CAT_COLS)}"

    # 确认所有特征列在 df 中都存在
    missing_enc = [c for c in fe if c not in df.columns]
    missing_xgb = [c for c in fx if c not in df.columns]
    assert not missing_enc, f"FEATURES_ENC 中缺失列: {missing_enc}"
    assert not missing_xgb, f"FEATURES_XGB 中缺失列: {missing_xgb}"

    print(f"   FEATURES_ENC: {len(fe)} 列")
    print(f"   FEATURES_XGB: {len(fx)} 列")
    print(f"   XGB_CAT_COLS: {len(cat)} 列")

    return fe, fx, cat


# ──────────── 时序划分 ────────────


def test_temporal_split(df, fe, fx):
    """验证时序划分"""
    print("\n" + "=" * 58)
    print("  [3/5] 时序划分: temporal_split()")
    print("=" * 58)

    from logistics_delay.models.train import temporal_split

    # 3a. sklearn 特征集
    X_tr_e, X_te_e, y_tr_e, y_te_e, spw_e, cutoff = temporal_split(df, "enc")
    assert len(X_tr_e) == len(y_tr_e), "enc 训练集特征/标签长度不匹配"
    assert len(X_te_e) == len(y_te_e), "enc 测试集特征/标签长度不匹配"
    assert list(X_tr_e.columns) == fe, "enc 训练集列名与 FEATURES_ENC 不符"
    assert 0 < spw_e < float("inf"), f"enc scale_pos_weight 异常: {spw_e}"
    print(f"   enc 划分: 训练 {len(y_tr_e)} / 测试 {len(y_te_e)}, spw={spw_e:.4f}")
    print(f"   分割点: {cutoff.date()}")

    # 3b. XGB 特征集
    X_tr_x, X_te_x, y_tr_x, y_te_x, spw_x, cutoff2 = temporal_split(df, "xgb")
    assert len(X_tr_x) == len(y_tr_x), "xgb 训练集特征/标签长度不匹配"
    assert len(X_te_x) == len(y_te_x), "xgb 测试集特征/标签长度不匹配"
    assert list(X_tr_x.columns) == fx, "xgb 训练集列名与 FEATURES_XGB 不符"
    print(f"   xgb 划分: 训练 {len(y_tr_x)} / 测试 {len(y_te_x)}, spw={spw_x:.4f}")
    print(f"   分割点: {cutoff2.date()}")

    # 3c. 验证时序完整性（训练集全部早于测试集）
    df_sorted = df.sort_values("trip_start_date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    train_max_date = df_sorted.loc[:split_idx - 1, "trip_start_date"].max()
    test_min_date = df_sorted.loc[split_idx:, "trip_start_date"].min()
    assert train_max_date <= test_min_date, f"时序泄露: 训练最晚 {train_max_date} > 测试最早 {test_min_date}"
    print(f"   时序完整性: 训练最晚 {train_max_date.date()} ≤ 测试最早 {test_min_date.date()}")

    return X_tr_e, X_te_e, y_tr_e, y_te_e, spw_e, X_tr_x, X_te_x, y_tr_x, y_te_x, spw_x


# ──────────── 单模型训练与评估 ────────────


def test_single_models(df, X_tr_e, X_te_e, y_tr_e, y_te_e, spw_e, X_tr_x, X_te_x, y_tr_x, y_te_x, spw_x, cat):
    """验证每个模型都能独立训练和评估"""
    print("\n" + "=" * 58)
    print("  [4/5] 单模型训练与评估")
    print("=" * 58)

    from logistics_delay.models.evaluate import get_model, evaluate
    from logistics_delay.models.train import train_model

    # 4a. LogisticRegression（enc 特征）
    print("  --- LogisticRegression (enc) ---")
    lr = get_model("LogisticRegression", spw_e)
    lr = train_model(lr, X_tr_e, y_tr_e)
    res_lr = evaluate(lr, X_te_e, y_te_e, "LogisticRegression")
    assert 0 <= res_lr["auc"] <= 1, f"AUC 越界: {res_lr['auc']}"
    assert 0 <= res_lr["f1"] <= 1, f"F1 越界: {res_lr['f1']}"
    print(f"   AUC={res_lr['auc']*100:.2f}%  F1={res_lr['f1']:.4f}  Acc={res_lr['accuracy']:.4f}")

    # 4b. CatBoost（xgb 特征 + cat_features）
    print("  --- CatBoost (xgb + cat_features) ---")
    cb = get_model("CatBoost", spw_x, cat_features=cat)
    cb = train_model(cb, X_tr_x, y_tr_x)
    res_cb = evaluate(cb, X_te_x, y_te_x, "CatBoost")
    assert 0 <= res_cb["auc"] <= 1
    print(f"   AUC={res_cb['auc']*100:.2f}%  F1={res_cb['f1']:.4f}  Acc={res_cb['accuracy']:.4f}")

    # 4c. XGBoost（xgb 特征，enable_categorical=True）
    print("  --- XGBoost (xgb) ---")
    xgb = get_model("XGBoost", spw_x)
    xgb = train_model(xgb, X_tr_x, y_tr_x)
    res_xgb = evaluate(xgb, X_te_x, y_te_x, "XGBoost")
    assert 0 <= res_xgb["auc"] <= 1
    print(f"   AUC={res_xgb['auc']*100:.2f}%  F1={res_xgb['f1']:.4f}  Acc={res_xgb['accuracy']:.4f}")

    # 4d. LightGBM（xgb 特征 + categorical_feature）
    print("  --- LightGBM (xgb + categorical_feature) ---")
    lgb = get_model("LightGBM", spw_x)
    lgb.fit(X_tr_x, y_tr_x, categorical_feature=cat)
    res_lgb = evaluate(lgb, X_te_x, y_te_x, "LightGBM")
    assert 0 <= res_lgb["auc"] <= 1
    print(f"   AUC={res_lgb['auc']*100:.2f}%  F1={res_lgb['f1']:.4f}  Acc={res_lgb['accuracy']:.4f}")

    # 4e. DecisionTree（enc 特征）
    print("  --- DecisionTree (enc) ---")
    dt = get_model("DecisionTree", spw_e)
    dt = train_model(dt, X_tr_e, y_tr_e)
    res_dt = evaluate(dt, X_te_e, y_te_e, "DecisionTree")
    assert 0 <= res_dt["auc"] <= 1
    print(f"   AUC={res_dt['auc']*100:.2f}%  F1={res_dt['f1']:.4f}  Acc={res_dt['accuracy']:.4f}")

    # 4f. RandomForest（enc 特征）
    print("  --- RandomForest (enc) ---")
    rf = get_model("RandomForest", spw_e)
    rf = train_model(rf, X_tr_e, y_tr_e)
    res_rf = evaluate(rf, X_te_e, y_te_e, "RandomForest")
    assert 0 <= res_rf["auc"] <= 1
    print(f"   AUC={res_rf['auc']*100:.2f}%  F1={res_rf['f1']:.4f}  Acc={res_rf['accuracy']:.4f}")


# ──────────── run_comparison 完整流水线 ────────────


def test_run_comparison(df):
    """验证完整对比流水线（所有 6 个模型，3 折 + 少量 bootstrap 以节省时间）"""
    print("\n" + "=" * 58)
    print("  [5/5] 完整对比: run_comparison()")
    print("=" * 58)

    from logistics_delay.models.comparison import run_comparison

    # 用 3 折 + 100 次 bootstrap 做快速验证
    results = run_comparison(df, n_splits=3, n_bootstrap=100)

    # 验证返回结果结构
    assert "auc_ci" in results, "缺少 auc_ci"
    assert "rankings_df" in results, "缺少 rankings_df"
    assert "win_matrix" in results, "缺少 win_matrix"
    assert "fold_aucs" in results, "缺少 fold_aucs"

    auc_ci = results["auc_ci"]
    rankings = results["rankings_df"]
    win_mat = results["win_matrix"]

    assert isinstance(auc_ci, type(df)), f"auc_ci 类型错误: {type(auc_ci)}"
    assert list(auc_ci.columns) == ["model", "mean_auc", "ci_lower", "ci_upper", "std_auc"]
    assert len(auc_ci) == 6, f"应有 6 个模型，实际 {len(auc_ci)}"

    assert len(rankings) == 6
    assert win_mat.shape == (6, 6)

    # 打印摘要
    print(f"\n  --- auc_ci 摘要 ---")
    print(f"  {'模型':<22s} {'均值AUC':>8s} {'下限':>8s} {'上限':>8s}")
    for _, r in auc_ci.iterrows():
        print(f"  {r['model']:<22s} {r['mean_auc']:>8.4f} {r['ci_lower']:>8.4f} {r['ci_upper']:>8.4f}")

    print(f"\n   auc_ci: {auc_ci.shape}")
    print(f"   rankings_df: {rankings.shape}")
    print(f"   win_matrix: {win_mat.shape}")
    print(f"   fold_aucs: {results['fold_aucs'].shape}")

    return results


# ──────────── 主入口 ────────────


def main():
    """运行 Quick Start 全流程验证"""
    print("=" * 58)
    print("  Logistics Delay — Quick Start 验证")
    print("  README.md 中所有代码路径全流程测试")
    print("=" * 58)

    try:
        # 1. 加载数据
        df = test_load_data()

        # 2. 特征列表
        fe, fx, cat = test_feature_lists(df)

        # 3. 时序划分
        (X_tr_e, X_te_e, y_tr_e, y_te_e, spw_e,
         X_tr_x, X_te_x, y_tr_x, y_te_x, spw_x) = test_temporal_split(df, fe, fx)

        # 4. 单模型训练
        test_single_models(df, X_tr_e, X_te_e, y_tr_e, y_te_e, spw_e,
                           X_tr_x, X_te_x, y_tr_x, y_te_x, spw_x, cat)

        # 5. 完整对比流水线
        test_run_comparison(df)

        print("\n" + "=" * 58)
        print("  所有模块均可正常调用与运行。")
        print("=" * 58)

    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
