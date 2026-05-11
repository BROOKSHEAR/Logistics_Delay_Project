# Logistics Delay Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red.svg)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green.svg)](https://lightgbm.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow.svg)](https://catboost.ai)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

Truck delivery delay prediction using ensemble machine learning, with temporal cross-validation, hyperparameter tuning, and SHAP model interpretation.

---

### 🇬🇧 English

This project predicts whether a truck delivery will be delayed based on information known **at the time of order creation** — transportation distance, vehicle type, origin/destination locations, and temporal features (weekday, month). No future-leaking features (e.g. actual trip duration) are used.

**Pipeline overview:**

- **6 models compared**: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost
- **Temporal cross-validation**: TimeSeriesSplit (5-fold) to respect chronological order
- **Two-stage hyperparameter tuning**: RandomizedSearchCV coarse search → GridSearchCV fine search
- **SHAP interpretation**: Global and local feature importance for tree-based models
- **Feature ablation**: Leave-one-out analysis to quantify each feature's contribution

---

### 🇳🇱 Nederlands

Dit project voorspelt of een vrachtwagenlevering vertraging oploopt, uitsluitend op basis van informatie die **op het moment van ordercreatie** bekend is — transportafstand, voertuigtype, herkomst/bestemming en temporele kenmerken (weekdag, maand). Er worden geen toekomstgegevens (zoals werkelijke reisduur) gebruikt.

**Overzicht van de aanpak:**

- **6 modellen vergeleken**: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost
- **Temporele kruisvalidatie**: TimeSeriesSplit (5-voud) met behoud van chronologische volgorde
- **Tweetraps hyperparameteroptimalisatie**: RandomizedSearchCV grof → GridSearchCV fijn
- **SHAP-analyse**: Globale en lokale feature-importance voor boommodellen
- **Feature-ablatie**: Leave-one-out-analyse om elke feature-bijdrage te kwantificeren

---

## Installation

```bash
# Install from source (editable, recommended for development)

git clone https://github.com/BROOKSHEAR/Logistics_Delay_Project
cd logistics_delay_project
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from logistics_delay.data.loader import load_processed
from logistics_delay.models.train import temporal_split
from logistics_delay.models.comparison import run_comparison

# 1. Load pre-processed feature data
df = load_processed()

# 2. Run 5-fold temporal cross-validation with all 6 models
results = run_comparison(df, n_bootstrap=2000)

# 3. View AUC confidence intervals
print(results["auc_ci"])

# 4. View model rankings across folds
print(results["rankings_df"])

# 5. View pairwise win matrix
print(results["win_matrix"])
```

For a single train/test split and model evaluation:

```python
from logistics_delay.models.train import temporal_split
from logistics_delay.models.evaluate import get_model, evaluate
from logistics_delay.features.engineering import FEATURES_XGB, XGB_CAT_COLS

X_train, X_test, y_train, y_test, spw, cutoff = temporal_split(df, "xgb")
model = get_model("CatBoost", spw, cat_features=XGB_CAT_COLS)
model.fit(X_train, y_train)
result = evaluate(model, X_test, y_test, "CatBoost")
print(f"AUC: {result['auc']:.4f}  F1: {result['f1']:.4f}")
```

## Project Structure

```
src/logistics_delay/
├── data/                  # Data loading & cleaning
│   ├── loader.py          # load_raw_data(), load_processed()
│   └── cleaner.py         # Conflict resolution, missing value imputation
├── features/              # Feature engineering
│   ├── engineering.py     # engineer_features(), get_feature_lists()
│   └── distance_fill_geo.py  # Geographic distance imputation
├── models/                # Training, evaluation, tuning, comparison
│   ├── train.py           # random_split(), temporal_split()
│   ├── evaluate.py        # evaluate(), get_model()
│   ├── tuning.py          # Two-stage hyperparameter search
│   └── comparison.py      # TimeSeriesSplit + Bootstrap CI
├── interpretation/        # Model explainability
│   └── shap_analysis.py   # SHAP beeswarm & bar plots
├── ablation/              # Feature & geographic ablation
│   └── ablation.py        # Leave-one-out feature ablation
└── utils/                 # Configuration
    └── paths.py           # Project paths, constants, seeds

notebooks/
├── 01_eda.ipynb           # Exploratory data analysis
├── 02_features.ipynb      # Feature engineering walkthrough
├── 03_modeling.ipynb      # Default-parameter model training
├── 04_tuning.ipynb        # Hyperparameter tuning + default vs tuned comparison
├── 05_ablation_viz.ipynb  # Feature ablation visualisation
└── 06_shap.ipynb          # Model comparison + SHAP analysis

tests/
├── test_geo_logic.py      # Unit tests for geographic logic
└── test_quick_start.py    # Quick Start integration tests

```

## Dataset

- **Source**: ~6,900 truck delivery records from Indian logistics operations (2019–2020)
- **Target**: Binary classification — delay vs on-time
- **Features**: Transportation distance, vehicle type, GPS provider, origin/destination codes, customer ID, weekday/month of departure, planned delivery window, market/regular flag
- **Key constraint**: Only features known at order creation time are used (no temporal leakage)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome. This project is part of an ongoing bachelor's thesis in Supply Chain Data Engineering.

- Found a bug? Open an issue.
- Want to improve a feature? Submit a pull request.
- Questions or suggestions? Feel free to start a discussion.
