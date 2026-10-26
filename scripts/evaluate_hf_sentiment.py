"""Evaluate the configured hosted sentiment model on a manually labeled CSV.

Input columns: text,label. Labels must be positive, neutral, or negative.
The script never prints or logs review text.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

from app.config import settings
from app.services.sentiment_service import SentimentService

LABELS = ["negative", "neutral", "positive"]


async def evaluate(input_path: Path, output_path: Path):
    dataset_bytes = input_path.read_bytes()
    with input_path.open(newline="", encoding="utf-8") as dataset_file:
        rows = list(csv.DictReader(dataset_file))
    if not rows or not {"text", "label"}.issubset(rows[0]):
        raise ValueError("Evaluation CSV must contain nonempty text and label columns")
    for row in rows:
        row["label"] = row["label"].strip().lower()
        if row["label"] not in LABELS or not row["text"].strip():
            raise ValueError("Each row requires nonempty text and a supported label")

    service = SentimentService()
    predictions = []
    for row in rows:
        result = await service.analyze(row["text"])
        predictions.append(result["label"])

    expected = [row["label"] for row in rows]
    report = classification_report(
        expected, predictions, labels=LABELS, output_dict=True, zero_division=0
    )
    output = {
        "model": settings.HF_SENTIMENT_MODEL,
        "provider": "huggingface_hf-inference",
        "evaluation_date_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "sample_count": len(rows),
        "metrics": report,
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(
            expected, predictions, labels=LABELS
        ).tolist(),
        "limitations": "Metrics apply only to this manually labeled sample; confidence is not calibrated correctness.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if os.getenv("SENTIMENT_EVALUATION_MLFLOW", "false").lower() == "true":
        import mlflow

        mlflow.set_experiment("InsightCart Sentiment Evaluation")
        with mlflow.start_run(run_name="huggingface-sentiment-evaluation"):
            mlflow.log_param("model", settings.HF_SENTIMENT_MODEL)
            mlflow.log_param("dataset_sha256", output["evaluation_dataset_sha256"])
            mlflow.log_param("sample_count", len(rows))
            for label in LABELS:
                for metric in ("precision", "recall", "f1-score"):
                    mlflow.log_metric(
                        f"{label}_{metric.replace('-', '_')}", report[label][metric]
                    )
            mlflow.log_metric("accuracy", report["accuracy"])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output", default=Path("artifacts/sentiment-evaluation.json"), type=Path
    )
    args = parser.parse_args()
    result = asyncio.run(evaluate(args.input, args.output))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sample_count": result["sample_count"],
                "model": result["model"],
            }
        )
    )


if __name__ == "__main__":
    main()
