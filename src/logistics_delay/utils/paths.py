"""
集中管理项目所有路径。

用法:
    from logistics_delay.utils.paths import DATA_RAW, check_data_exists, ...
"""
from __future__ import annotations

from pathlib import Path

# ── 项目根目录 ──────────────────────────────────
# 此文件位于 src/logistics_delay/utils/paths.py
# 向上 4 层 → 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── 数据路径 ────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw" / "Delivery truck trip data.xlsx"
DATA_PROCESSED = DATA_DIR / "processed"

# ── 输出路径 ────────────────────────────────────
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_EDA = FIGURES_DIR / "eda"
FIGURES_SHAP = FIGURES_DIR / "shap"
FIGURES_MODELS = FIGURES_DIR / "models"

TABLES_DIR = OUTPUTS_DIR / "tables"
TABLES_METRICS = TABLES_DIR / "metrics"



# ── 模型路径 ────────────────────────────────────
MODELS_DIR = OUTPUTS_DIR / "models"

# ── 全局配置 ────────────────────────────────────
SEED = 42
RANDOM_STATE = 42


def check_data_exists() -> Path:
    """检查原始数据文件是否存在，不存在则报错退出。

    Returns:
        数据文件的绝对路径。

    Raises:
        FileNotFoundError: 数据文件不存在。
    """
    if not DATA_RAW.exists():
        raise FileNotFoundError(
            f"原始数据文件不存在: {DATA_RAW}\n"
            f"请确认文件位于 {DATA_RAW}，"
            f"或从 {DATA_DIR / 'raw'} 目录检查。"
        )
    return DATA_RAW


def ensure_output_dirs() -> None:
    """确保所有输出目录存在（自动创建）。"""
    for d in [FIGURES_EDA, FIGURES_SHAP, FIGURES_MODELS,
              TABLES_METRICS, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
