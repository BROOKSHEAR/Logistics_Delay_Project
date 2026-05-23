"""
Centralized project path management.

Usage:
    from logistics_delay.utils.paths import DATA_RAW, check_data_exists, ...
"""
from __future__ import annotations

from pathlib import Path

# ── Project root ──────────────────────────────────
# This file is at src/logistics_delay/utils/paths.py
# Go up 4 levels → project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── Data paths ────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw" / "Delivery truck trip data.xlsx"
DATA_PROCESSED = DATA_DIR / "processed"

# ── Output paths ──────────────────────────────────
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_EDA = FIGURES_DIR / "eda"
FIGURES_SHAP = FIGURES_DIR / "shap"
FIGURES_MODELS = FIGURES_DIR / "models"

TABLES_DIR = OUTPUTS_DIR / "tables"

# ── Global config ─────────────────────────────────
SEED = 42
RANDOM_STATE = 42


def check_data_exists() -> Path:
    """Check that the raw data file exists, exit with error if not.

    Returns:
        Absolute path to the data file.

    Raises:
        FileNotFoundError: Data file not found.
    """
    if not DATA_RAW.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {DATA_RAW}\n"
            f"Please verify the file is at {DATA_RAW},"
            f"or check the {DATA_DIR / 'raw'} directory."
        )
    return DATA_RAW


def ensure_output_dirs() -> None:
    """Ensure all output directories exist (auto-create)."""
    for d in [FIGURES_EDA, FIGURES_SHAP, FIGURES_MODELS]:
        d.mkdir(parents=True, exist_ok=True)
