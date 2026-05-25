#!/usr/bin/env python3
"""
Orchestrator: run all heavy computation (tuning + analysis) sequentially.

Usage:  python -m logistics_delay.run_all

Equivalent to running:
    python -m logistics_delay.run_tuning
    python -m logistics_delay.run_analysis
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

if __name__ == "__main__":
    print("=" * 60)
    print("  Run ALL — tuning + analysis")
    print("=" * 60)

    # ── 1. Tuning (two-stage + Optuna) ──
    print("\n>>> Running tuning...")
    from logistics_delay import run_tuning
    run_tuning.main()

    # ── 2. Analysis (ablation + comparison + SHAP) ──
    print("\n>>> Running analysis...")
    from logistics_delay import run_analysis
    run_analysis.main()

    print("\n" + "=" * 60)
    print("  ALL DONE — tuning + analysis complete")
    print("=" * 60)
