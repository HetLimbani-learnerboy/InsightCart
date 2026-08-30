"""
Project : InsightCart

File : model_gate.py

Purpose :
Evaluate candidate model against baseline thresholds before promotion.
"""


def model_quality_gate(
    new_metrics: dict,
    production_metrics: dict = None,
    minimum_f1: float = 0.90,
) -> bool:
    """Decide whether a newly trained model can be promoted."""
    new_f1 = float(new_metrics.get("f1", 0.0))
    production_f1 = (
        float(production_metrics.get("f1", 0.0)) if production_metrics else 0.0
    )

    print(f"New Model F1       : {new_f1:.4f}")
    print(f"Production Model F1: {production_f1:.4f}")

    if new_f1 < minimum_f1:
        print(f"Model rejected: F1 {new_f1:.4f} < minimum {minimum_f1:.4f}")
        return False

    if production_metrics and new_f1 <= production_f1:
        print(
            f"Model rejected: new F1 {new_f1:.4f} <= production F1 {production_f1:.4f}"
        )
        return False

    print("Model passed quality gate.")
    return True
