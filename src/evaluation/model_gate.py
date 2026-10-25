"""
Project : InsightCart

File : model_gate.py

Purpose :
Evaluate candidate model against baseline thresholds before promotion.
"""

import math
import os

METRICS = ("accuracy", "precision", "recall", "f1")


def gate_thresholds() -> dict:
    """Read technical defaults from environment; product thresholds need approval."""
    return {
        metric: float(os.getenv(f"MODEL_MIN_{metric.upper()}", "0"))
        for metric in METRICS
    }


def evaluate_model_quality(
    candidate: dict,
    production: dict = None,
    thresholds: dict = None,
    primary_metric: str = None,
    minimum_improvement: float = None,
) -> tuple:
    """Return approval and a reason; compare only identical held-out evaluations."""
    thresholds = thresholds or gate_thresholds()
    primary_metric = (primary_metric or os.getenv("MODEL_PRIMARY_METRIC", "f1")).lower()
    minimum_improvement = (
        float(os.getenv("MODEL_MINIMUM_IMPROVEMENT", "0"))
        if minimum_improvement is None
        else float(minimum_improvement)
    )
    if primary_metric not in METRICS:
        return False, f"unsupported primary metric: {primary_metric}"
    if not isinstance(candidate, dict) or not isinstance(production, dict):
        return False, "candidate and production metrics are required"
    candidate_eval_id = candidate.get("evaluation_dataset_id")
    production_eval_id = production.get("evaluation_dataset_id")
    if not candidate_eval_id or candidate_eval_id != production_eval_id:
        return False, "candidate and production evaluation dataset identities differ"

    values = {}
    for metric in METRICS:
        try:
            candidate_value = float(candidate[metric])
            production_value = float(production[metric])
            threshold = float(thresholds[metric])
        except (KeyError, TypeError, ValueError):
            return False, f"missing or invalid {metric} metric/threshold"
        if not all(
            math.isfinite(value)
            for value in (candidate_value, production_value, threshold)
        ):
            return False, f"non-finite {metric} metric/threshold"
        if not 0 <= threshold <= 1:
            return False, f"{metric} threshold must be between 0 and 1"
        if not 0 <= candidate_value <= 1 or not 0 <= production_value <= 1:
            return False, f"{metric} metric must be between 0 and 1"
        if candidate_value < threshold:
            return (
                False,
                f"candidate {metric} {candidate_value:.4f} is below {threshold:.4f}",
            )
        values[metric] = (candidate_value, production_value)

    candidate_primary, production_primary = values[primary_metric]
    if not math.isfinite(minimum_improvement) or not 0 <= minimum_improvement <= 1:
        return False, "minimum improvement must be between 0 and 1"
    if candidate_primary <= production_primary or candidate_primary < (
        production_primary + minimum_improvement
    ):
        return False, (
            f"candidate {primary_metric} improvement is below the configured minimum"
        )
    return True, "candidate passed configured metrics on the matching evaluation set"


def model_quality_gate(
    new_metrics: dict,
    production_metrics: dict = None,
    minimum_f1: float = None,
) -> bool:
    """Boolean compatibility wrapper for callers that only need pass/fail."""
    thresholds = gate_thresholds()
    if minimum_f1 is not None:
        thresholds["f1"] = float(minimum_f1)
    return evaluate_model_quality(new_metrics, production_metrics, thresholds)[0]
