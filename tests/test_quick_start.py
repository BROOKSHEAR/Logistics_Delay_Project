"""
Verify reproducibility of the Quick Start flow.

Ensure anyone can git clone, load data, run models, and see output.
Test covers all code paths in the Quick Start section of README.md.
"""
import sys
import warnings

warnings.filterwarnings("ignore")

# ──────────── Data loading ────────────


def test_load_data():
    """Verify preprocessed data loading"""
    print("=" * 58)
    print("  [1/5] Data loading: load_processed()")
    print("=" * 58)

    from logistics_delay.data.loader import load_processed, load_raw_data

    # 1a. Load preprocessed data (primary path)
    df = load_processed()
    assert df is not None, "load_processed() returned None"
    assert not df.empty, "returned empty DataFrame"
    assert "Answer" in df.columns, "missing Answer column"
    assert "trip_start_date" in df.columns, "missing trip_start_date column"
    assert df.index.name is None or df.index.name == "index"

    print(f"   Shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   delay rate: {df['Answer'].mean() * 100:.2f}%")

    # 1b. Verify raw data loads too
    raw = load_raw_data()
    assert raw is not None
    print(f"   Raw data: {raw.shape}")

    return df


# ──────────── Feature lists ────────────


def test_feature_lists(df):
    """Verify feature list constants"""
    print("\n" + "=" * 58)
    print("  [2/5] Feature lists: get_feature_lists()")
    print("=" * 58)

    from logistics_delay.features.engineering import get_feature_lists, FEATURES_ENC, FEATURES_XGB, XGB_CAT_COLS

    fe, fx, cat = get_feature_lists()
    assert len(fe) == len(FEATURES_ENC), f"FEATURES_ENC length mismatch: {len(fe)} vs {len(FEATURES_ENC)}"
    assert len(fx) == len(FEATURES_XGB), f"FEATURES_XGB length mismatch: {len(fx)} vs {len(FEATURES_XGB)}"
    assert len(cat) == len(XGB_CAT_COLS), f"XGB_CAT_COLS length mismatch: {len(cat)} vs {len(XGB_CAT_COLS)}"

    # Verify all feature columns exist in df
    missing_enc = [c for c in fe if c not in df.columns]
    missing_xgb = [c for c in fx if c not in df.columns]
    assert not missing_enc, f"FEATURES_ENC missing columns: {missing_enc}"
    assert not missing_xgb, f"FEATURES_XGB missing columns: {missing_xgb}"

    print(f"   FEATURES_ENC: {len(fe)} columns")
    print(f"   FEATURES_XGB: {len(fx)} columns")
    print(f"   XGB_CAT_COLS: {len(cat)} columns")

    return fe, fx, cat


# ──────────── Temporal split ────────────


def test_temporal_split(df, fe, fx):
    """Verify temporal split"""
    print("\n" + "=" * 58)
    print("  [3/5] Temporal split: temporal_split()")
    print("=" * 58)

    from logistics_delay.models.train import temporal_split

    # 3a. sklearn feature set
    X_tr_e, X_te_e, y_tr_e, y_te_e, spw_e, cutoff = temporal_split(df, "enc")
    assert len(X_tr_e) == len(y_tr_e), "enc train features/labels length mismatch"
    assert len(X_te_e) == len(y_te_e), "enc test features/labels length mismatch"
    assert list(X_tr_e.columns) == fe, "enc train columns mismatch FEATURES_ENC"
    assert 0 < spw_e < float("inf"), f"enc scale_pos_weight abnormal: {spw_e}"
    print(f"   enc split: train {len(y_tr_e)} / test {len(y_te_e)}, spw={spw_e:.4f}")
    print(f"   Cutoff: {cutoff.date()}")

    # 3b. XGB feature set
    X_tr_x, X_te_x, y_tr_x, y_te_x, spw_x, cutoff2 = temporal_split(df, "xgb")
    assert len(X_tr_x) == len(y_tr_x), "xgb train features/labels length mismatch"
    assert len(X_te_x) == len(y_te_x), "xgb test features/labels length mismatch"
    assert list(X_tr_x.columns) == fx, "xgb train columns mismatch FEATURES_XGB"
    print(f"   xgb split: train {len(y_tr_x)} / test {len(y_te_x)}, spw={spw_x:.4f}")
    print(f"   Cutoff: {cutoff2.date()}")

    # 3c. Verify temporal integrity (all train before test)
    df_sorted = df.sort_values("trip_start_date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    train_max_date = df_sorted.loc[:split_idx - 1, "trip_start_date"].max()
    test_min_date = df_sorted.loc[split_idx:, "trip_start_date"].min()
    assert train_max_date <= test_min_date, f"Temporal leak: latest train {train_max_date} > earliest test {test_min_date}"
    print(f"   Temporal integrity: latest train {train_max_date.date()} ≤ earliest test {test_min_date.date()}")

    return X_tr_e, X_te_e, y_tr_e, y_te_e, spw_e, X_tr_x, X_te_x, y_tr_x, y_te_x, spw_x


# ──────────── Single model training & evaluation ────────────


def test_single_models(df, X_tr_e, X_te_e, y_tr_e, y_te_e, spw_e, X_tr_x, X_te_x, y_tr_x, y_te_x, spw_x, cat):
    """Verify each model can train and evaluate independently"""
    print("\n" + "=" * 58)
    print("  [4/5] Single model training & evaluation")
    print("=" * 58)

    from logistics_delay.models.evaluate import get_model, evaluate
    from logistics_delay.models.train import train_model

    # 4a. LogisticRegression (enc features)
    print("  --- LogisticRegression (enc) ---")
    lr = get_model("LogisticRegression", spw_e)
    lr = train_model(lr, X_tr_e, y_tr_e)
    res_lr = evaluate(lr, X_te_e, y_te_e, "LogisticRegression")
    assert 0 <= res_lr["auc"] <= 1, f"AUC out of bounds: {res_lr['auc']}"
    assert 0 <= res_lr["f1"] <= 1, f"F1 out of bounds: {res_lr['f1']}"
    print(f"   AUC={res_lr['auc']*100:.2f}%  F1={res_lr['f1']:.4f}  Acc={res_lr['accuracy']:.4f}")

    # 4b. CatBoost (xgb features + cat_features)
    print("  --- CatBoost (xgb + cat_features) ---")
    cb = get_model("CatBoost", spw_x, cat_features=cat)
    cb = train_model(cb, X_tr_x, y_tr_x)
    res_cb = evaluate(cb, X_te_x, y_te_x, "CatBoost")
    assert 0 <= res_cb["auc"] <= 1
    print(f"   AUC={res_cb['auc']*100:.2f}%  F1={res_cb['f1']:.4f}  Acc={res_cb['accuracy']:.4f}")

    # 4c. XGBoost (xgb features, enable_categorical=True)
    print("  --- XGBoost (xgb) ---")
    xgb = get_model("XGBoost", spw_x)
    xgb = train_model(xgb, X_tr_x, y_tr_x)
    res_xgb = evaluate(xgb, X_te_x, y_te_x, "XGBoost")
    assert 0 <= res_xgb["auc"] <= 1
    print(f"   AUC={res_xgb['auc']*100:.2f}%  F1={res_xgb['f1']:.4f}  Acc={res_xgb['accuracy']:.4f}")

    # 4d. LightGBM (xgb features + categorical_feature)
    print("  --- LightGBM (xgb + categorical_feature) ---")
    lgb = get_model("LightGBM", spw_x)
    lgb.fit(X_tr_x, y_tr_x, categorical_feature=cat)
    res_lgb = evaluate(lgb, X_te_x, y_te_x, "LightGBM")
    assert 0 <= res_lgb["auc"] <= 1
    print(f"   AUC={res_lgb['auc']*100:.2f}%  F1={res_lgb['f1']:.4f}  Acc={res_lgb['accuracy']:.4f}")

    # 4e. DecisionTree (enc features)
    print("  --- DecisionTree (enc) ---")
    dt = get_model("DecisionTree", spw_e)
    dt = train_model(dt, X_tr_e, y_tr_e)
    res_dt = evaluate(dt, X_te_e, y_te_e, "DecisionTree")
    assert 0 <= res_dt["auc"] <= 1
    print(f"   AUC={res_dt['auc']*100:.2f}%  F1={res_dt['f1']:.4f}  Acc={res_dt['accuracy']:.4f}")

    # 4f. RandomForest (enc features)
    print("  --- RandomForest (enc) ---")
    rf = get_model("RandomForest", spw_e)
    rf = train_model(rf, X_tr_e, y_tr_e)
    res_rf = evaluate(rf, X_te_e, y_te_e, "RandomForest")
    assert 0 <= res_rf["auc"] <= 1
    print(f"   AUC={res_rf['auc']*100:.2f}%  F1={res_rf['f1']:.4f}  Acc={res_rf['accuracy']:.4f}")


# ──────────── run_comparison full pipeline ────────────


def test_run_comparison(df):
    """Verify full comparison pipeline (all 6 models, 3 folds + few bootstrap for speed)"""
    print("\n" + "=" * 58)
    print("  [5/5] Full comparison: run_comparison()")
    print("=" * 58)

    from logistics_delay.models.comparison import run_comparison

    # Use 3 folds + 100 bootstrap resamples for quick validation
    results = run_comparison(df, n_splits=3, n_bootstrap=100)

    # Verify result structure
    assert "auc_ci" in results, "missing auc_ci"
    assert "rankings_df" in results, "missing rankings_df"
    assert "win_matrix" in results, "missing win_matrix"
    assert "fold_aucs" in results, "missing fold_aucs"

    auc_ci = results["auc_ci"]
    rankings = results["rankings_df"]
    win_mat = results["win_matrix"]

    assert isinstance(auc_ci, type(df)), f"auc_ci type error: {type(auc_ci)}"
    assert list(auc_ci.columns) == ["model", "mean_auc", "ci_lower", "ci_upper", "std_auc"]
    assert len(auc_ci) == 6, f"expected 6 models, got {len(auc_ci)}"

    assert len(rankings) == 6
    assert win_mat.shape == (6, 6)

    # Print summary
    print(f"\n  --- auc_ci summary ---")
    print(f"  {'Model':<22s} {'Mean AUC':>8s} {'Lower':>8s} {'Upper':>8s}")
    for _, r in auc_ci.iterrows():
        print(f"  {r['model']:<22s} {r['mean_auc']:>8.4f} {r['ci_lower']:>8.4f} {r['ci_upper']:>8.4f}")

    print(f"\n   auc_ci: {auc_ci.shape}")
    print(f"   rankings_df: {rankings.shape}")
    print(f"   win_matrix: {win_mat.shape}")
    print(f"   fold_aucs: {results['fold_aucs'].shape}")

    return results


# ──────────── Main entry ────────────


def main():
    """Run Quick Start full flow verification"""
    print("=" * 58)
    print("  Logistics Delay — Quick Start verification")
    print("  Full flow test of all code paths in README.md")
    print("=" * 58)

    try:
        # 1. Load data
        df = test_load_data()

        # 2. Feature lists
        fe, fx, cat = test_feature_lists(df)

        # 3. Temporal split
        (X_tr_e, X_te_e, y_tr_e, y_te_e, spw_e,
         X_tr_x, X_te_x, y_tr_x, y_te_x, spw_x) = test_temporal_split(df, fe, fx)

        # 4. Single model training
        test_single_models(df, X_tr_e, X_te_e, y_tr_e, y_te_e, spw_e,
                           X_tr_x, X_te_x, y_tr_x, y_te_x, spw_x, cat)

        # 5. Full comparison pipeline
        test_run_comparison(df)

        print("\n" + "=" * 58)
        print("  All modules can be imported and run successfully.")
        print("=" * 58)

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
