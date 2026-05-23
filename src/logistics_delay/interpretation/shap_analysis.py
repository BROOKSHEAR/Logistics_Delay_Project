"""
SHAP interpretability analysis module.

Provides one-shot SHAP computation, importance rankings, SHAP force plot generation,
and correlation with ablation experiments.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def compute_shap_importance(model, X_test: pd.DataFrame,
                            feature_names: list[str] | None = None,
                            model_type: str = "tree") -> pd.DataFrame:
    """Compute SHAP values and return feature importance ranking DataFrame.

    Args:
        model: Trained model (supports tree / linear / kernel explainer).
        X_test: Test feature matrix.
        feature_names: Feature name list, defaults to ``X_test.columns``.
        model_type: ``"tree"``, ``"linear"`` or ``"kernel"``.

    Returns:
        DataFrame sorted by ``mean_abs_shap`` descending.
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X_test)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_test)

    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        # For multi-class, take positive class (index 1)
        shap_values = shap_values[1]

    feat_names = feature_names or list(X_test.columns)
    imp_df = pd.DataFrame({
        "feature": feat_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(f"\n[shap] SHAP importance ranking:")
    for i, (_, row) in enumerate(imp_df.iterrows()):
        print(f"  {i + 1}. {row['feature']:35s}  {row['mean_abs_shap']:.6f}")

    return imp_df, shap_values, explainer


def compare_with_ablation(shap_df: pd.DataFrame,
                          ablation_df: pd.DataFrame,
                          split_name: str = "random") -> float:
    """Compute Pearson correlation between SHAP importance ranking and ablation AUC drop.

    High correlation means SHAP and ablation agree,
    increasing confidence in the feature importance ranking.

    Args:
        shap_df: DataFrame from ``compute_shap_importance``.
        ablation_df: Ablation result with ``removed_feat`` and ``auc_drop`` columns.
        split_name: Split name (used only for printing).

    Returns:
        Pearson correlation coefficient.
    """
    merged = pd.merge(
        shap_df,
        ablation_df[["removed_feat", "auc_drop"]],
        left_on="feature",
        right_on="removed_feat",
        how="left",
    )
    pearson_r = merged["mean_abs_shap"].corr(merged["auc_drop"].abs())

    print(f"\n[shap] SHAP vs ablation Pearson correlation ({split_name}): {pearson_r:.4f}")
    return pearson_r


BUSINESS_MAP = {
    'planned_days_enc': 'Planned days binned',
    'TRANSPORTATION_DISTANCE_IN_KM': 'Distance (km)',
    'GpsProvider': 'GPS Provider',
    'origin_city': 'Origin city',
    'start_month': 'Start month',
    'customerID': 'Customer ID',
    'Minimum_kms_to_be_covered_in_a_day': 'Min daily km',
    'DestinationLocation_Code': 'Dest code',
    'booking_prefix': 'Booking prefix',
    'start_weekday': 'Start weekday',
    'vehicleType': 'Vehicle type',
    'OriginLocation_Code': 'Origin code',
    'is_market': 'Is market',
    'supplier_is_large': 'Is large supplier',
    'dest_city': 'Dest city',
}


def render_force_plot(shap_values_1d, base_value, feature_values, feature_names,
                      title, save_path, business_map=None):
    """Render ``shap.force_plot`` with ``#1``-``#N`` labels and anti-overlap layout.

    Replaces original feature-name labels with numbered labels to avoid crowding,
    adjusts text y-positions so ``E[f(x)]`` and feature numbers do not overlap,
    and enlarges the figure canvas slightly for breathing room.

    Args:
        shap_values_1d: 1-D array of SHAP values for a single sample.
        base_value: Expected value (``explainer.expected_value``).
        feature_values: Feature values for the sample (Series or array-like).
        feature_names: Ordered list of feature names matching SHAP dims.
        title: Plot title string.
        save_path: Output file path (stem used; ``_new.pdf`` and ``_new.png`` saved).
        business_map: Optional dict mapping raw feature names to display names.

    Returns:
        list[dict]: Ordered blocks with keys ``number``, ``display_name``, ``value``, ``shap``.
    """
    sv = np.asarray(shap_values_1d, dtype=float)

    fv_list = (feature_values.tolist() if hasattr(feature_values, 'tolist')
               else list(feature_values))
    order = np.argsort(np.abs(sv))[::-1]
    blocks = []
    for fi in order:
        raw_name = feature_names[fi]
        blocks.append({
            'shap': float(sv[fi]),
            'display_name': business_map.get(raw_name, raw_name) if business_map else raw_name,
            'value': fv_list[fi],
        })
    pos_b = sorted([b for b in blocks if b['shap'] >= 0],
                   key=lambda b: b['shap'], reverse=True)
    neg_b = sorted([b for b in blocks if b['shap'] < 0],
                   key=lambda b: b['shap'])
    pc = float(base_value)
    for b in pos_b:
        b['x_left'] = pc; b['x_right'] = pc + b['shap']
        b['x_center'] = (b['x_left'] + b['x_right']) / 2.0; pc = b['x_right']
    nc = float(base_value)
    for b in neg_b:
        b['x_left'] = nc + b['shap']; b['x_right'] = nc
        b['x_center'] = (b['x_left'] + b['x_right']) / 2.0; nc = b['x_left']
    all_b = neg_b + pos_b
    blocks_by_pos = sorted(all_b, key=lambda b: b['x_center'])
    for i, b in enumerate(blocks_by_pos):
        b['number'] = i + 1

    shap.force_plot(base_value, sv, feature_values, matplotlib=True, show=False)
    fig = plt.gcf()
    ax = plt.gca()

    # Move E[f(x)] and f(x) UP to free space
    for t in ax.texts:
        txt = t.get_text()
        if 'E[' in txt or ('f(' in txt and 'E[' not in txt):
            x, y = t.get_position()
            t.set_position((x, y + 0.12))

    # Replace feature labels with #N and move DOWN
    feature_texts = []
    for t in ax.texts:
        txt = t.get_text()
        if ('E[' in txt or 'f(' in txt
                or 'higher' in txt.lower() or 'lower' in txt.lower()):
            continue
        if '=' in txt:
            feature_texts.append(t)
    feature_texts.sort(key=lambda t: t.get_position()[0])
    for i, t in enumerate(feature_texts):
        t.set_text(f'#{i+1}')
        x, y = t.get_position()
        t.set_position((x, y - 0.18))

    ow, oh = fig.get_size_inches()
    fig.set_size_inches(ow, oh + 1.5)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    save_path = Path(save_path)
    fig.savefig(str(save_path.parent / f'{save_path.stem}.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(str(save_path.parent / f'{save_path.stem}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return blocks_by_pos


def generate_all_force_plots(data_path=None, output_dir=None):
    """Full pipeline: load data, train CatBoost, compute SHAP, generate 4 force plots.

    Args:
        data_path: Path to processed data. If ``None``, uses default loader path.
        output_dir: Output directory for figures. If ``None``, uses ``FIGURES_SHAP``.

    Returns:
        tuple: (catboost_model, shap_values, base_value, test_data, blocks_by_category)
    """
    import tempfile
    import warnings
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score
    from logistics_delay.data.loader import load_processed
    from logistics_delay.features.engineering import (
        XGB_CAT_COLS, FEATURES_XGB,
    )
    from logistics_delay.utils.paths import FIGURES_SHAP

    out = output_dir or FIGURES_SHAP
    warnings.filterwarnings('ignore')

    print('=' * 60)
    print('  Loading data ...')
    print('=' * 60)
    df = load_processed(data_path) if data_path else load_processed()
    print(f'  Data shape: {df.shape}')

    # Temporal split 80/20
    df_sorted = df.sort_values('trip_start_date').reset_index(drop=True)
    si = int(len(df_sorted) * 0.8)
    X_train = df_sorted.loc[:si-1][FEATURES_XGB].copy()
    X_test  = df_sorted.loc[si:][FEATURES_XGB].copy()
    y_train = df_sorted.loc[:si-1, 'Answer'].reset_index(drop=True)
    y_test  = df_sorted.loc[si:, 'Answer'].reset_index(drop=True)
    test_ids = df_sorted.loc[si:, 'BookingID'].reset_index(drop=True)

    for c in XGB_CAT_COLS:
        if c in X_train.columns:
            X_train[c] = X_train[c].astype('category')
        if c in X_test.columns:
            X_test[c] = X_test[c].astype('category')

    spw = (y_train == 0).sum() / (y_train == 1).sum()
    print(f'  Train: {len(X_train)}  Test: {len(X_test)}  spw={spw:.4f}')

    # Train CatBoost
    print()
    print('=' * 60)
    print('  Training CatBoost ...')
    print('=' * 60)
    params = {
        'learning_rate': 0.05, 'depth': 8, 'iterations': 100,
        'l2_leaf_reg': 5, 'border_count': 64,
        'bagging_temperature': 1, 'random_strength': 0,
        'random_seed': 42, 'verbose': 0,
        'train_dir': tempfile.gettempdir(),
        'class_weights': {0: 1.0, 1: spw},
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, cat_features=XGB_CAT_COLS)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f'  CatBoost AUC (temporal): {auc*100:.2f}%')

    # SHAP values
    print()
    print('=' * 60)
    print('  Computing SHAP values ...')
    print('=' * 60)
    explainer = shap.TreeExplainer(model)
    shap_v = explainer.shap_values(X_test)
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1]

    # Classification breakdown
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    tp = (y_test.values == 1) & (y_pred == 1)
    tn = (y_test.values == 0) & (y_pred == 0)
    fp = (y_test.values == 0) & (y_pred == 1)
    fn = (y_test.values == 1) & (y_pred == 0)
    print(f'  Classification: TP={tp.sum()}  TN={tn.sum()}  FP={fp.sum()}  FN={fn.sum()}')

    def _pick_rep(mask, prefer_delay=True):
        idx_arr = np.where(mask)[0]
        if len(idx_arr) == 0:
            return None
        total = shap_v[idx_arr].sum(axis=1)
        return idx_arr[np.argmax(total) if prefer_delay else np.argmin(total)]

    cats = [
        ('TP - Correct delay', 'True Positive: Actual Delayed / Pred Delayed',
         _pick_rep(tp, True), 'force_plot_tp'),
        ('TN - Correct on-time', 'True Negative: Actual On-time / Pred On-time',
         _pick_rep(tn, False), 'force_plot_tn'),
        ('FP - False delay', 'False Positive: Actual On-time / Pred Delayed',
         _pick_rep(fp, True), 'force_plot_fp'),
        ('FN - False on-time', 'False Negative: Actual Delayed / Pred On-time',
         _pick_rep(fn, False), 'force_plot_fn'),
    ]

    # Generate force plots
    print()
    print('=' * 60)
    print('  Generating force plots ...')
    print('=' * 60)
    results = {}
    for subtitle, title, idx, fname in cats:
        if idx is None:
            print(f'  [Skip] {subtitle} -- no samples')
            results[subtitle] = None
            continue

        print(f'  Plotting {subtitle} (idx={idx})...')
        blocks = render_force_plot(
            shap_values_1d=shap_v[idx],
            base_value=base_val,
            feature_values=X_test.iloc[idx],
            feature_names=FEATURES_XGB,
            title=title,
            save_path=Path(out) / fname,
            business_map=BUSINESS_MAP,
        )

        print(f'  BookingID: {test_ids[idx]}')
        print(f'  Pred delay prob: {y_pred_proba[idx]:.3f}  |  '
              f'Actual: {"Delayed" if y_test.values[idx] else "On-time"}')
        print()
        print('  Feature Reference:\n')
        for b in blocks:
            n, name, val, sval = b['number'], b['display_name'], b['value'], b['shap']
            arrow = ('Push up' if sval > 0.01
                     else ('Pull down' if sval < -0.01 else '~'))
            print(f'    #{n:2d}  {name:<22s}  = {str(val):<12s}  SHAP {sval:+.4f}  {arrow}')
        print()
        print('-' * 80)
        print()
        results[subtitle] = {'idx': idx, 'blocks': blocks, 'booking_id': test_ids[idx],
                             'pred': y_pred_proba[idx], 'actual': y_test.values[idx]}

    print('=' * 60)
    print('  Done!')
    print('=' * 60)
    print(f'  CatBoost AUC={auc*100:.2f}%')

    return model, shap_v, base_val, (X_test, y_test, test_ids), results
