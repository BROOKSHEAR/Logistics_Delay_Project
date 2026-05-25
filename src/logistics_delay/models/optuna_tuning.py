"""
Optuna-based hyperparameter tuning module.

Replaces the two-stage (RandomizedSearchCV → GridSearchCV) approach with
Optuna's TPE sampler + MedianPruner for more efficient hyperparameter search.
Parameter spaces are expanded vs the two-stage approach.

Usage:
    from logistics_delay.models.optuna_tuning import (
        run_optuna_search,
        suggest_lr_params, suggest_dt_params,
        suggest_rf_params, suggest_xgb_params,
        suggest_lgbm_params, suggest_cb_params,
    )
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Callable, Optional

import pickle

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score

from logistics_delay.utils.paths import SEED, TABLES_DIR

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
#  Optuna suggest functions — expanded parameter spaces
# ════════════════════════════════════════════════════════════════

def suggest_lr_params(trial):
    """LogisticRegression params via Optuna — continuous C + all legal penalty/solver combos."""
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])

    # Enforce solver constraint: l1 and elasticnet require saga
    if penalty in ("l1", "elasticnet"):
        solver = "saga"

    params = {
        "penalty": penalty,
        "solver": solver,
        "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "max_iter": trial.suggest_int("max_iter", 500, 5000),
    }

    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.95)

    return params


def suggest_dt_params(trial):
    """DecisionTree params via Optuna — expanded space (+ccp_alpha, +min_impurity_decrease)."""
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.05),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.01),
    }


def suggest_rf_params(trial):
    """RandomForest params via Optuna — expanded space (+ccp_alpha, +min_impurity_decrease, float max_features).

    Note: ``_mf_type`` / ``_mf_val`` are underscore-prefixed meta-params sampled by
    Optuna but not passed to the model constructor. ``run_optuna_search`` strips
    keys starting with ``_`` and reconstructs ``max_features`` from them during retrain.
    """
    # Underscore-prefixed = Optuna-only meta-params (not passed to model constructor)
    mf_type = trial.suggest_categorical("_mf_type", ["sqrt", "log2", "float"])
    max_features = (
        trial.suggest_float("_mf_val", 0.3, 0.8)
        if mf_type == "float"
        else mf_type
    )

    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "max_features": max_features,
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.05),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.01),
    }


def suggest_xgb_params(trial, spw_candidates=None):
    """XGBoost params via Optuna — expanded space (+max_delta_step, +grow_policy, continuous ranges)."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 100, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 100, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
    }
    if spw_candidates is not None:
        params["scale_pos_weight"] = trial.suggest_categorical("scale_pos_weight", spw_candidates)
    return params


def suggest_lgbm_params(trial, spw_candidates=None):
    """LightGBM params via Optuna — expanded space (+min_gain_to_split, +bagging_freq, +feature_fraction)."""
    # max_depth=-1 is "no limit". Use categorical to avoid TPE wasting budget near -1/0.
    _unlimited = trial.suggest_categorical("_unlimited_depth", [True, False])
    max_depth = -1 if _unlimited else trial.suggest_int("max_depth", 2, 20)

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 300, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
        "max_depth": max_depth,
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 100, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 100, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 5.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
    }
    if spw_candidates is not None:
        params["scale_pos_weight"] = trial.suggest_categorical("scale_pos_weight", spw_candidates)
    return params


def suggest_cb_params(trial):
    """CatBoost params via Optuna — expanded parameter space (+bootstrap_type)."""
    # bootstrap_type governs which of bagging_temperature / subsample is active.
    # Bayesian → bagging_temperature; Bernoulli/MVS → subsample; no need to search both.
    bt = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "depth": trial.suggest_int("depth", 2, 12),
        "iterations": trial.suggest_int("iterations", 50, 1000, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "leaf_estimation_method": trial.suggest_categorical("leaf_estimation_method", ["Newton", "Gradient"]),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": bt,
    }

    if bt == "Bayesian":
        # bagging_temperature only applies for Bayesian bootstrap
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
    elif bt in ("Bernoulli", "MVS"):
        # subsample only applies for Bernoulli / MVS bootstrap
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    return params


# ════════════════════════════════════════════════════════════════
#  Objective builder
# ════════════════════════════════════════════════════════════════

def _make_objective(model_class, suggest_fn, X_train, y_train, n_splits,
                    fit_params, model_init_kwargs, early_stopping_rounds):
    """Build Optuna objective: sample params → CV AUC (with pruning per fold).

    If ``early_stopping_rounds > 0``, passes ``eval_set`` to .fit() for
    XGBClassifier / LGBMClassifier / CatBoostClassifier so each fold stops
    early when validation AUC stagnates.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model_name = model_class.__name__
    _can_early_stop = model_name in ("XGBClassifier", "LGBMClassifier", "CatBoostClassifier")

    def objective(trial):
        params = suggest_fn(trial)
        # model_init_kwargs first, so suggest_fn results take priority
        merged = {**(model_init_kwargs or {}), **params}
        model = model_class(**merged)

        cv_aucs = []
        for step, (tr_idx, val_idx) in enumerate(tscv.split(X_train, y_train)):
            X_tr = X_train.iloc[tr_idx] if hasattr(X_train, "iloc") else X_train[tr_idx]
            y_tr = y_train.iloc[tr_idx] if hasattr(y_train, "iloc") else y_train[tr_idx]
            X_val = X_train.iloc[val_idx] if hasattr(X_train, "iloc") else X_train[val_idx]
            y_val = y_train.iloc[val_idx] if hasattr(y_train, "iloc") else y_train[val_idx]

            try:
                kwargs = dict(**(fit_params or {}))
                es = _can_early_stop and early_stopping_rounds and early_stopping_rounds > 0

                if es and model_name == "CatBoostClassifier":
                    m = clone(model)
                    kwargs["eval_set"] = (X_val, y_val)
                    kwargs["early_stopping_rounds"] = early_stopping_rounds
                elif es and model_name == "LGBMClassifier":
                    # LightGBM 4.0+: early_stopping_rounds in constructor
                    m = model_class(**merged, early_stopping_rounds=early_stopping_rounds)
                    kwargs["eval_set"] = [(X_val, y_val)]
                elif es and model_name == "XGBClassifier":
                    # XGBoost 3.x: eval_set goes in .fit(), no per-fold early stopping
                    m = clone(model)
                else:
                    m = clone(model)

                m.fit(X_tr, y_tr, **kwargs)
                auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
                cv_aucs.append(auc)
            except Exception:
                import traceback
                traceback.print_exc()
                # Mark trial as failed — do not return 0.0 which pollutes TPE posterior
                raise optuna.TrialPruned()

            trial.report(auc, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(sum(cv_aucs) / max(len(cv_aucs), 1))

    return objective


# ════════════════════════════════════════════════════════════════
#  Main entry point
# ════════════════════════════════════════════════════════════════

def run_optuna_search(
    model_class: type,
    suggest_fn: Callable,
    X_train,
    y_train,
    X_test,
    y_test,
    n_trials: int = 200,
    n_splits: int = 5,
    direction: str = "maximize",
    fit_params: Optional[dict] = None,
    model_init_kwargs: Optional[dict] = None,
    early_stopping_rounds: Optional[int] = None,
    timeout: Optional[float] = None,
    seed: int = SEED,
    study_name: Optional[str] = None,
    save_study_path: Optional[str] = None,
) -> dict:
    """Optuna hyperparameter search with TPE sampler + MedianPruner.

    Parameters
    ----------
    model_class : type
        Unfitted model class (e.g. ``XGBClassifier``).
    suggest_fn : Callable
        Function ``(trial, **kwargs) -> dict`` that suggests hyperparameters.
    X_train, y_train, X_test, y_test : array-like
        Data splits.
    n_trials : int
        Number of Optuna trials (default 200).
    n_splits : int
        TimeSeriesSplit folds (default 5).
    direction : str
        ``"maximize"`` or ``"minimize"`` (default ``"maximize"``).
    fit_params : dict or None
        Extra kwargs passed to ``.fit()`` (e.g. ``cat_features``).
    model_init_kwargs : dict or None
        Fixed kwargs for model constructor (e.g. ``random_state``, ``verbosity``).
        These are overridden by any param with the same name from ``suggest_fn``.
    early_stopping_rounds : int or None
        If set, passes ``eval_set`` + ``early_stopping_rounds`` to ``.fit()``
        for XGB/LightGBM/CatBoost (ignored for other models).
    timeout : float or None
        Time limit in seconds for the study.
    seed : int
        Random seed for the TPE sampler.
    study_name : str or None
        Optional study name.

    Returns
    -------
    dict with keys: ``model``, ``cv_auc``, ``test_auc``, ``test_f1``,
    ``best_params``, ``n_trials``, ``n_pruned``.
    """
    model_name = model_class.__name__

    # ── Study ──
    sampler = TPESampler(seed=seed, n_startup_trials=10)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )

    objective = _make_objective(
        model_class, suggest_fn, X_train, y_train,
        n_splits, fit_params, model_init_kwargs, early_stopping_rounds,
    )

    # ── Run ──
    print(f"\n{'=' * 60}")
    print(f"  Optuna search: {model_name} ({n_trials} trials)")
    print(f"  Sampler: TPE | Pruner: Median | CV: {n_splits}-fold TSS")
    print(f"{'=' * 60}")

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    n_total = len(study.trials)
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    best_params = study.best_params

    print(f"\n  [Optuna] Best CV AUC = {study.best_value * 100:.2f}%")
    print(f"  Trials: {n_total} total, {n_pruned} pruned")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

    # ── Retrain best on full train & evaluate on test ──
    # Strip underscore-prefixed meta-params (Optuna-only, not for model constructor)
    model_params = {k: v for k, v in best_params.items() if not k.startswith("_")}

    # Fix LogisticRegression solver (may be wrong in best_params when penalty overrides it)
    if model_name == "LogisticRegression" and model_params.get("penalty") in ("l1", "elasticnet"):
        model_params["solver"] = "saga"

    # Reconstruct max_features for RandomForest (_mf_type/_mf_val in best_params)
    if model_name == "RandomForestClassifier":
        mf_type = best_params.get("_mf_type")
        if mf_type == "float":
            model_params["max_features"] = best_params.get("_mf_val", 0.5)
        elif mf_type in ("sqrt", "log2"):
            model_params["max_features"] = mf_type

    # model_init_kwargs first → model_params (from suggest) take priority
    merged_best = {**(model_init_kwargs or {}), **model_params}
    fit_kwargs = dict(**(fit_params or {}))

    _supports_retrain_es = early_stopping_rounds and early_stopping_rounds > 0 \
        and model_name in ("LGBMClassifier", "XGBClassifier", "CatBoostClassifier")

    if _supports_retrain_es:
        n_val = max(int(len(X_train) * 0.1), 1)
        X_tr = X_train.iloc[:-n_val] if hasattr(X_train, "iloc") else X_train[:-n_val]
        y_tr = y_train.iloc[:-n_val] if hasattr(y_train, "iloc") else y_train[:-n_val]
        X_val = X_train.iloc[-n_val:] if hasattr(X_train, "iloc") else X_train[-n_val:]
        y_val = y_train.iloc[-n_val:] if hasattr(y_train, "iloc") else y_train[-n_val:]

        if model_name == "LGBMClassifier":
            merged_best["early_stopping_rounds"] = early_stopping_rounds
            best_model = model_class(**merged_best)
            best_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], **(fit_params or {}))
        elif model_name == "XGBClassifier":
            # XGBoost 3.x: early_stopping_rounds in constructor, eval_set in .fit()
            merged_best["early_stopping_rounds"] = early_stopping_rounds
            best_model = model_class(**merged_best)
            best_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], **(fit_params or {}))
        else:
            # CatBoostClassifier: early_stopping_rounds + eval_set in .fit()
            best_model = model_class(**merged_best)
            fit_kwargs["eval_set"] = (X_val, y_val)
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            best_model.fit(X_tr, y_tr, **fit_kwargs)
    else:
        best_model = model_class(**merged_best)
        best_model.fit(X_train, y_train, **fit_kwargs)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    result = {
        "model": model_name,
        "cv_auc": study.best_value,
        "test_auc": roc_auc_score(y_test, y_prob),
        "test_f1": f1_score(y_test, y_pred),
        "best_params": best_params,
        "n_trials": n_total,
        "n_pruned": n_pruned,
    }

    print(f"\n  [OK] {model_name} complete")
    print(f"     CV AUC  = {result['cv_auc'] * 100:.2f}%")
    print(f"     Test AUC = {result['test_auc'] * 100:.2f}%")
    print(f"     Test F1  = {result['test_f1']:.4f}")

    if save_study_path:
        os.makedirs(os.path.dirname(save_study_path), exist_ok=True)
        with open(save_study_path, "wb") as f:
            pickle.dump(study, f)
        print(f"     Study saved → {save_study_path}")

    return result


# ════════════════════════════════════════════════════════════════
#  Save results
# ════════════════════════════════════════════════════════════════

def save_optuna_results(results_df, save_dir=None):
    """Save Optuna tuning results to CSV (includes best_params as readable string).

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame summarising dicts returned by ``run_optuna_search``.
    save_dir : str or Path
        Save directory (default ``TABLES_DIR``).
    """
    if save_dir is None:
        save_dir = TABLES_DIR
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "optuna_tuning_results.csv")
    df = results_df.copy()
    if "best_params" in df.columns:
        df["best_params"] = df["best_params"].apply(
            lambda d: ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in d.items()
            ) if isinstance(d, dict) else str(d)
        )
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Optuna results saved -> {path}")
