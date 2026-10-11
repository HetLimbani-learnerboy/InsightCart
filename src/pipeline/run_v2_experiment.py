"""Run a reproducible, non-promoting experiment on processed dataset v2."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import joblib
import pandas as pd

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
from src.evaluation.model_gate import model_quality_gate
from src.logger import logger


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


def run_experiment(dataset_path: str, output_dir: str, random_seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Starting isolated dataset-v2 experiment")

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
    evaluation = ModelEvaluation().initiate_model_evaluation(
        trainer_artifacts,
        transformed.X_test,
        transformed.y_test,
        metrics_path=evaluation_path,
        dataset_version="clean_dataset_v2",
        comparison_path=comparison_path,
        tracking_parameters={
            "dataset_path": dataset_path,
            "random_seed": random_seed,
            "validation_rows": len(transformed.y_validation),
            "test_rows": len(transformed.y_test),
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

    production_metrics_path = os.path.join(
        project_root, "artifacts/production_metrics.json"
    )
    try:
        with open(production_metrics_path, encoding="utf-8") as metrics_file:
            production_metrics = json.load(metrics_file)
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Production quality-gate baseline unavailable: %s", error)
        production_metrics = None
    gate_passed = model_quality_gate(
        {"f1": evaluation.metrics["F1 Score"]},
        production_metrics,
        minimum_f1=0.90,
    )

    metadata = {
        "dataset_path": dataset_path,
        "dataset_version": "clean_dataset_v2",
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
            "split": {"train": 0.60, "validation": 0.20, "test": 0.20},
            "hyperparameter_tuning": False,
            "preprocessing_fit_on": "training split only",
            "candidate_models": list(trainer.models),
        },
        "production_quality_gate_passed": gate_passed,
        "production_promotion_performed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
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
        default="data/processed/clean_dataset_v2.csv",
        help="CSV dataset path (default: processed dataset v2)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/experiments/v2",
        help="Experiment output directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    run_experiment(arguments.dataset, arguments.output_dir, arguments.seed)
