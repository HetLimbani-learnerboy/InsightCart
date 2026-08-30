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
mlflow.set_tracking_uri(f"sqlite:///{SQLITE_DB_PATH}")


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
            logger.info(
                f"Logging Evaluation Results to MLflow using SQLite DB at {SQLITE_DB_PATH}"
            )
            mlflow.set_tracking_uri(f"sqlite:///{SQLITE_DB_PATH}")
            mlflow.set_experiment(self.config.experiment_name)

            with mlflow.start_run(run_name=trainer_artifacts.best_model_name):
                mlflow.log_param("Algorithm", trainer_artifacts.best_model_name)
                mlflow.set_tag("selected_model", trainer_artifacts.best_model_name)
                mlflow.log_param("Calibration", "Sigmoid")

                for param_name, param_val in best_params.items():
                    mlflow.log_param(param_name, param_val)

                for metric_name, metric_val in metrics.items():
                    mlflow.log_metric(metric_name, metric_val)

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    skops_trusted_types=[
                        "sklearn.calibration._CalibratedClassifier",
                        "sklearn.calibration._SigmoidCalibration",
                        "sklearn.calibration.CalibratedClassifierCV",
                    ],
                )

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

            logger.info("MLflow Experiment Logging Completed")

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
