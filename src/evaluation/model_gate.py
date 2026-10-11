"""
Project : InsightCart

File : model_gate.py

Purpose :
Evaluate candidate model against baseline thresholds before promotion.
"""

import math


def model_quality_gate(
    new_metrics: dict,
    production_metrics: dict = None,
    minimum_f1: float = 0.90,
) -> bool:
    """Fail closed unless both candidate and production F1 are comparable."""
    try:
        new_f1 = float(new_metrics["f1"])
        production_f1 = float(production_metrics["f1"])
        if not math.isfinite(new_f1) or not math.isfinite(production_f1):
            raise ValueError("F1 scores must be finite numbers")
    except (KeyError, TypeError, ValueError):
        print("Model rejected: valid candidate and production F1 scores are required")
        return False

    print(f"New Model F1       : {new_f1:.4f}")
    print(f"Production Model F1: {production_f1:.4f}")

    if new_f1 < minimum_f1:
        print(f"Model rejected: F1 {new_f1:.4f} < minimum {minimum_f1:.4f}")
        return False

    if new_f1 < production_f1:
        print(
            f"Model rejected: new F1 {new_f1:.4f} < production F1 {production_f1:.4f}"
        )
        return False

    print("Model passed quality gate.")
    return True
