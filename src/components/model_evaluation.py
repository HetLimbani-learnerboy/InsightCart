"""
Project : InsightCart

File : model_evaluation.py

Purpose :
Evaluate trained models, compute metrics, produce reports, and log experiments to MLflow.
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.components.model_trainer import ModelTrainerArtifacts
from src.evaluation.evaluate_models import save_metrics
from src.exception import CustomException
from src.logger import logger

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
SQLITE_DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")
DEFAULT_TRACKING_URI = f"sqlite:///{SQLITE_DB_PATH}"
DEFAULT_ARTIFACT_ROOT = os.path.join(PROJECT_ROOT, "mlruns")

# Respect an externally provided MLFLOW_TRACKING_URI (CI sets this to a
# runner-local path). Fall back to a local sqlite store for local dev.
# IMPORTANT: this must NOT be overridden later inside the class — doing so
# silently ignores whatever the environment/CI configured.
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
mlflow.set_tracking_uri(TRACKING_URI)


def _artifact_uri_for(path: str) -> str:
    """Build an OS-correct file:// URI for a local directory."""
    os.makedirs(path, exist_ok=True)
    return f"file:{os.path.abspath(path)}"


def _artifact_location_is_usable(artifact_location: str) -> bool:
    """
    Check whether an experiment's recorded artifact_location can actually
    be written to from THIS machine. Non-local stores (s3://, etc.) are
    assumed usable and left alone.
    """
    if not artifact_location.startswith("file:"):
        return True

    local_path = artifact_location.replace("file://", "").replace("file:", "")
    try:
        os.makedirs(local_path, exist_ok=True)
        return True
    except (PermissionError, OSError):
        return False


def get_or_create_experiment(name: str, artifact_root: str = DEFAULT_ARTIFACT_ROOT):
    """
    Fetch an MLflow experiment by name, creating it if it doesn't exist.

    If the tracking DB already has an experiment with this name but its
    stored artifact_location was recorded on a *different* machine/OS
    (e.g. a Mac path picked up from a shared/committed mlflow.db), that
    location won't be writable here. Rather than crashing the whole
    training pipeline, we archive the stale experiment and create a
    fresh one pointed at a path valid on this machine.
    """
    experiment = mlflow.get_experiment_by_name(name)

    if experiment is None or experiment.lifecycle_stage == "deleted":
        exp_id = mlflow.create_experiment(
            name, artifact_location=_artifact_uri_for(artifact_root)
        )
        return mlflow.get_experiment(exp_id)

    if not _artifact_location_is_usable(experiment.artifact_location):
        logger.warning(
            f"Experiment '{name}' has an artifact_location "
            f"('{experiment.artifact_location}') that isn't usable on this "
            "machine. Archiving it and creating a fresh experiment instead."
        )
        client = mlflow.tracking.MlflowClient()
        client.rename_experiment(
            experiment.experiment_id, f"{name}_stale_{experiment.experiment_id}"
        )
        client.delete_experiment(experiment.experiment_id)
        exp_id = mlflow.create_experiment(
            name, artifact_location=_artifact_uri_for(artifact_root)
        )
        return mlflow.get_experiment(exp_id)

    return experiment


@dataclass
class ModelEvaluationConfig:
    experiment_name: str = "InsightCart AI Generated Review Detection"


@dataclass
class ModelEvaluationArtifacts:
    metrics: Dict[str, float]
    classification_report: str
    confusion_matrix: np.ndarray


class ModelEvaluation:

    def __init__(self):
        self.config = ModelEvaluationConfig()

    def initiate_model_evaluation(
        self,
        trainer_artifacts: ModelTrainerArtifacts,
        X_test: Any,
        y_test: Any,
    ) -> ModelEvaluationArtifacts:
        try:
            logger.info("Model Evaluation Started")

            model = trainer_artifacts.trained_model
            best_params = trainer_artifacts.best_parameters

            # --- 1. Predictions ---
            logger.info("Generating Predictions on Test Dataset")
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            # --- 2. Calculate Metrics ---
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            roc_auc = roc_auc_score(y_test, probabilities)

            metrics = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC AUC": roc_auc,
            }

            logger.info(f"Evaluation Metrics: {metrics}")

            # --- 3. Classification Report & Confusion Matrix ---
            report = classification_report(y_test, predictions)
            cm = confusion_matrix(y_test, predictions)

            logger.info("\nClassification Report:\n%s", report)
            logger.info("\nConfusion Matrix:\n%s", cm)

            # --- 4. MLflow Logging ---
            # NOTE: tracking URI is set once at module import time, honoring
            # MLFLOW_TRACKING_URI if the environment (e.g. CI) provided one.
            # We do NOT re-set it here, and we never let MLflow logging
            # failures take down the whole pipeline — metrics/report/cm are
            # already computed above and are what downstream steps need.
            try:
                logger.info(f"Logging Evaluation Results to MLflow at {TRACKING_URI}")
                experiment = get_or_create_experiment(self.config.experiment_name)
                mlflow.set_experiment(experiment_id=experiment.experiment_id)

                with mlflow.start_run(run_name=trainer_artifacts.best_model_name):
                    mlflow.log_param("Algorithm", trainer_artifacts.best_model_name)
                    mlflow.set_tag("selected_model", trainer_artifacts.best_model_name)
                    mlflow.log_param("Calibration", "Sigmoid")

                    for param_name, param_val in best_params.items():
                        mlflow.log_param(param_name, param_val)

                    for metric_name, metric_val in metrics.items():
                        mlflow.log_metric(metric_name, metric_val)

                    try:
                        mlflow.sklearn.log_model(
                            sk_model=model,
                            name="model",
                            skops_trusted_types=[
                                "sklearn.calibration._CalibratedClassifier",
                                "sklearn.calibration._SigmoidCalibration",
                                "sklearn.calibration.CalibratedClassifierCV",
                            ],
                        )
                    except (PermissionError, OSError) as artifact_err:
                        logger.warning(
                            "Skipping MLflow model artifact logging "
                            f"(non-fatal): {artifact_err}"
                        )

                logger.info("MLflow Experiment Logging Completed")
            except Exception as mlflow_err:
                # MLflow tracking is observability, not the pipeline's core
                # output. Log and continue rather than failing the whole run.
                logger.warning(f"MLflow logging failed (non-fatal): {mlflow_err}")

            # --- 5. Save metrics to artifacts for downstream consumption ---
            try:
                metrics_out = {
                    "accuracy": float(metrics["Accuracy"]),
                    "precision": float(metrics["Precision"]),
                    "recall": float(metrics["Recall"]),
                    "f1": float(metrics["F1 Score"]),
                    "roc_auc": float(metrics["ROC AUC"]),
                }
                saved_path = save_metrics(
                    metrics_out, out_path="artifacts/evaluation_results.json"
                )
                logger.info(f"Saved evaluation metrics to: {saved_path}")
            except Exception:
                logger.warning("Failed to save evaluation metrics to artifacts")

            evaluation_artifacts = ModelEvaluationArtifacts(
                metrics=metrics,
                classification_report=report,
                confusion_matrix=cm,
            )

            logger.info("Model Evaluation Completed Successfully")
            return evaluation_artifacts

        except Exception as e:
            logger.error("Model Evaluation Failed")
            raise CustomException(e, sys)
