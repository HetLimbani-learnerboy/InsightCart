"""Run a reproducible, non-promoting experiment on processed dataset v2."""

import argparse
import hashlib
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from src.components.data_validation import DataValidation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.model_pusher import ModelPusher
from src.constants import (
    CATEGORY_COLUMN,
    ENCODER_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RATING_COLUMN,
    SCALER_PATH,
    TARGET_COLUMN,
    TFIDF_PATH,
)
from src.evaluation.model_gate import evaluate_model_quality
from src.logger import logger
from src.utils.common import load_object
from src.utils.preprocessing import clean_text


def evaluate_production_model(evaluation_dataframe: pd.DataFrame) -> dict:
    """Evaluate production artifacts on the promotion holdout, not the final test set."""
    bundle_path = os.path.join(MODEL_DIR, "production_model_bundle.pkl")
    if os.path.exists(bundle_path):
        bundle = load_object(bundle_path)
        model = bundle["model"]
        vectorizer = bundle["tfidf"]
        encoder = bundle["category_encoder"]
        scaler = bundle["rating_scaler"]
        model_version = bundle.get("model_version", bundle.get("model_name", "unknown"))
    else:
        model = load_object(MODEL_PATH)
        vectorizer = load_object(TFIDF_PATH)
        encoder = load_object(ENCODER_PATH)
        scaler = load_object(SCALER_PATH)
        model_version = os.path.splitext(os.path.basename(MODEL_PATH))[0]

    reviews = evaluation_dataframe["text_"].map(clean_text)
    text_features = vectorizer.transform(reviews)
    category_features = encoder.transform(evaluation_dataframe[[CATEGORY_COLUMN]])
    rating_features = scaler.transform(evaluation_dataframe[[RATING_COLUMN]])
    features = hstack([text_features, category_features, rating_features])
    predictions = model.predict(features)
    labels = evaluation_dataframe[TARGET_COLUMN]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, average="binary", zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, average="binary", zero_division=0)
        ),
        "f1": float(f1_score(labels, predictions, average="binary", zero_division=0)),
        "evaluation_dataset_id": None,
        "model_version": model_version,
    }


def package_versions():
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "xgboost",
        "lightgbm",
        "mlflow",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def run_experiment(
    dataset_path: str,
    output_dir: str,
    random_seed: int = 42,
    promote: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Starting isolated dataset-v2 experiment")
    with open(dataset_path, "rb") as dataset_file:
        dataset_fingerprint = hashlib.sha256(dataset_file.read()).hexdigest()

    dataframe = DataIngestion(
        DataIngestionConfig(raw_data_path=dataset_path)
    ).initiate_data_ingestion()
    dataframe = DataValidation().validate_dataset(dataframe)

    preprocessing_dir = os.path.join(output_dir, "preprocessing")
    os.makedirs(preprocessing_dir, exist_ok=True)
    transformation = DataTransformation(
        DataTransformationConfig(
            tfidf_path=os.path.join(preprocessing_dir, "tfidf_vectorizer.pkl"),
            encoder_path=os.path.join(preprocessing_dir, "category_encoder.pkl"),
            scaler_path=os.path.join(preprocessing_dir, "rating_scaler.pkl"),
            random_state=random_seed,
        )
    )
    transformed = transformation.initiate_data_transformation(dataframe)

    comparison_path = os.path.join(output_dir, "model_comparison.csv")
    trainer = ModelTrainer(
        ModelTrainerConfig(
            random_state=random_seed,
            comparison_path=comparison_path,
        )
    )
    trainer_artifacts = trainer.initiate_model_training(
        transformed.X_train,
        transformed.X_validation,
        transformed.y_train,
        transformed.y_validation,
    )

    evaluation_path = os.path.join(output_dir, "evaluation_results.json")
    production_metrics = evaluate_production_model(transformed.promotion_dataframe)
    production_metrics["evaluation_dataset_id"] = transformed.promotion_dataset_id
    candidate_promotion_predictions = trainer_artifacts.trained_model.predict(
        transformed.X_promotion
    )
    candidate_promotion_metrics = {
        "accuracy": float(
            accuracy_score(transformed.y_promotion, candidate_promotion_predictions)
        ),
        "precision": float(
            precision_score(
                transformed.y_promotion,
                candidate_promotion_predictions,
                average="binary",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                transformed.y_promotion,
                candidate_promotion_predictions,
                average="binary",
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                transformed.y_promotion,
                candidate_promotion_predictions,
                average="binary",
                zero_division=0,
            )
        ),
        "evaluation_dataset_id": transformed.promotion_dataset_id,
    }
    evaluation = ModelEvaluation().initiate_model_evaluation(
        trainer_artifacts,
        transformed.X_test,
        transformed.y_test,
        metrics_path=evaluation_path,
        dataset_version=dataset_fingerprint,
        evaluation_dataset_id=transformed.test_dataset_id,
        comparison_path=comparison_path,
        tracking_parameters={
            "dataset_path": dataset_path,
            "dataset_fingerprint": dataset_fingerprint,
            "previous_production_model_version": production_metrics["model_version"],
            "random_seed": random_seed,
            "validation_rows": len(transformed.y_validation),
            "promotion_dataset_id": transformed.promotion_dataset_id,
            "promotion_rows": len(transformed.y_promotion),
            "test_rows": len(transformed.y_test),
        },
        tracking_metrics={
            **{
                f"production_promotion_{metric}": production_metrics[metric]
                for metric in ("accuracy", "precision", "recall", "f1")
            },
            **{
                f"candidate_promotion_{metric}": candidate_promotion_metrics[metric]
                for metric in ("accuracy", "precision", "recall", "f1")
            },
        },
    )

    model_path = os.path.join(output_dir, "selected_model.pkl")
    joblib.dump(
        {
            "model": trainer_artifacts.trained_model,
            "tfidf": transformed.tfidf,
            "category_encoder": transformed.category_encoder,
            "rating_scaler": transformed.rating_scaler,
            "model_name": trainer_artifacts.best_model_name,
        },
        model_path,
    )

    gate_passed, gate_reason = evaluate_model_quality(
        {
            **candidate_promotion_metrics,
        },
        production_metrics,
    )
    promotion_performed = False
    if promote and gate_passed:
        promoted_metrics = {
            "accuracy": float(evaluation.metrics["Accuracy"]),
            "precision": float(evaluation.metrics["Precision"]),
            "recall": float(evaluation.metrics["Recall"]),
            "f1": float(evaluation.metrics["F1 Score"]),
            "roc_auc": (
                float(evaluation.metrics["ROC AUC"])
                if pd.notna(evaluation.metrics["ROC AUC"])
                else None
            ),
            "evaluation_dataset_id": transformed.promotion_dataset_id,
            "dataset_version": dataset_fingerprint,
            "model_version": dataset_fingerprint[:12],
            "selected_model": trainer_artifacts.best_model_name,
            "promotion_decision": "approved",
            "promotion_reason": "candidate passed configured promotion holdout gate",
            "previous_production_model_version": production_metrics["model_version"],
        }
        ModelPusher().initiate_model_pusher(
            trainer_artifacts,
            evaluation_metrics=promoted_metrics,
            transformation_artifacts=transformed,
            model_version=dataset_fingerprint[:12],
        )
        promotion_performed = True
        gate_reason = "candidate passed validation and explicit promotion completed"
    elif promote:
        gate_reason = f"promotion blocked: {gate_reason}"

    metadata = {
        "dataset_path": os.path.basename(dataset_path),
        "dataset_version": dataset_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "promotion_dataset_id": transformed.promotion_dataset_id,
        "test_dataset_id": transformed.test_dataset_id,
        "dataset_shape": list(dataframe.shape),
        "columns": dataframe.columns.tolist(),
        "target_column": "label",
        "target_distribution": {
            str(key): int(value)
            for key, value in dataframe["label"].value_counts().items()
        },
        "missing_values": {
            column: int(count) for column, count in dataframe.isna().sum().items()
        },
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "excluded_feature_columns": [
            column
            for column in dataframe.columns
            if column not in {"text_", "category", "rating", "label"}
        ],
        "selected_model": trainer_artifacts.best_model_name,
        "primary_selection_metric": "validation F1; ROC-AUC tie-break",
        "validation_metrics": trainer_artifacts.validation_metrics,
        "test_metrics": {
            key: float(value) if pd.notna(value) else None
            for key, value in evaluation.metrics.items()
        },
        "training_configuration": {
            "random_seed": random_seed,
            "split": {
                "train": 0.60,
                "validation": 0.10,
                "promotion": 0.10,
                "test": 0.20,
            },
            "hyperparameter_tuning": False,
            "preprocessing_fit_on": "training split only",
            "candidate_models": list(trainer.models),
        },
        "production_quality_gate_passed": gate_passed,
        "production_quality_gate_reason": gate_reason,
        "promotion_metrics": candidate_promotion_metrics,
        "previous_production_model_version": production_metrics["model_version"],
        "production_promotion_metrics": {
            key: production_metrics[key]
            for key in ("accuracy", "precision", "recall", "f1")
        },
        "production_promotion_performed": promotion_performed,
        "library_versions": package_versions(),
        "artifacts": {
            "comparison": comparison_path,
            "evaluation": evaluation_path,
            "selected_model": model_path,
        },
    }
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, allow_nan=False)

    if getattr(evaluation, "run_id", None):
        try:
            import mlflow

            client = mlflow.tracking.MlflowClient()
            client.set_tag(
                evaluation.run_id,
                "promotion_decision",
                "promoted"
                if promotion_performed
                else "rejected"
                if promote
                else "candidate_only",
            )
            client.set_tag(evaluation.run_id, "promotion_reason", gate_reason)
            client.set_tag(
                evaluation.run_id,
                "candidate_model_version",
                dataset_fingerprint[:12],
            )
        except Exception:
            logger.exception("Could not tag MLflow run with promotion decision")
            if promotion_performed:
                logger.warning(
                    "Model was promoted; decision metadata is preserved in %s",
                    metadata_path,
                )

    logger.info(
        "Dataset-v2 experiment finished: selected=%s validation_f1=%.4f test_f1=%.4f gate=%s",
        trainer_artifacts.best_model_name,
        trainer_artifacts.validation_metrics["F1 Score"],
        evaluation.metrics["F1 Score"],
        gate_passed,
    )
    return trainer_artifacts, evaluation, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/raw/fake_reviews_dataset.csv",
        help="CSV dataset path (default: DVC-tracked raw dataset)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/experiments/dvc-candidate",
        help="Experiment output directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote only when the configured gate passes against matching production metrics",
    )
    arguments = parser.parse_args()
    run_experiment(
        arguments.dataset,
        arguments.output_dir,
        arguments.seed,
        promote=arguments.promote,
    )
