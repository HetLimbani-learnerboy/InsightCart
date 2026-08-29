def model_quality_gate(
    new_metrics,
    production_metrics,
    minimum_f1=0.90,
):
    """Decide whether a newly trained model can be promoted.

    Rules:
    1. New model must have F1 >= minimum_f1.
    2. If production_metrics provided, new model must have F1 > production F1.
       (ties are rejected to avoid unnecessary replacements.)
    """
    new_f1 = float(new_metrics.get("f1", 0.0))
    production_f1 = float(production_metrics.get("f1", 0.0)) if production_metrics else 0.0

    print(f"New Model F1       : {new_f1:.4f}")
    print(f"Production Model F1: {production_f1:.4f}")

    # Rule 1: minimum acceptable quality
    if new_f1 < minimum_f1:
        print(f"Model rejected: F1 {new_f1:.4f} < minimum {minimum_f1:.4f}")
        return False

    # Rule 2: must be strictly better than production if production exists
    if production_metrics and new_f1 <= production_f1:
        print(f"Model rejected: new F1 {new_f1:.4f} <= production F1 {production_f1:.4f}")
        return False

    print("Model passed quality gate.")
    return True
