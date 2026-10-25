"""Train candidate classifiers and select by validation performance."""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from src.exception import CustomException
from src.logger import logger


@dataclass
class ModelTrainerConfig:
    random_state: int = 42
    selection_metric: str = "F1 Score"
    model_names: list = field(
        default_factory=lambda: [
            "Logistic Regression",
            "Multinomial Naive Bayes",
            "Decision Tree",
            "Random Forest",
            "Linear SVM",
            "XGBoost",
            "LightGBM",
        ]
    )
    comparison_path: str = "artifacts/experiments/manual-candidate/model_comparison.csv"


@dataclass
class ModelTrainerArtifacts:
    trained_model: Any
    best_model_name: str
    comparison_dataframe: pd.DataFrame
    best_parameters: Dict[str, Any]
    trained_models: Dict[str, Any] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = None):
        self.config = config or ModelTrainerConfig()
        seed = self.config.random_state
        self.models = {
            "Logistic Regression": LogisticRegression(
                random_state=seed, max_iter=1000, class_weight="balanced"
            ),
            "Multinomial Naive Bayes": MultinomialNB(),
            "Decision Tree": DecisionTreeClassifier(
                random_state=seed, class_weight="balanced"
            ),
            "Random Forest": RandomForestClassifier(
                random_state=seed,
                n_estimators=200,
                n_jobs=-1,
                class_weight="balanced",
            ),
            "Linear SVM": LinearSVC(
                random_state=seed,
                max_iter=10000,
                dual="auto",
                class_weight="balanced",
            ),
        }
        self.dependency_errors = {}
        try:
            from xgboost import XGBClassifier

            self.models["XGBoost"] = XGBClassifier(
                random_state=seed, eval_metric="logloss", n_jobs=-1
            )
        except Exception as error:
            self.models["XGBoost"] = None
            self.dependency_errors["XGBoost"] = f"XGBoost unavailable: {error}"
        try:
            from lightgbm import LGBMClassifier

            self.models["LightGBM"] = LGBMClassifier(
                random_state=seed, verbose=-1, class_weight="balanced"
            )
        except Exception as error:
            self.models["LightGBM"] = None
            self.dependency_errors["LightGBM"] = f"LightGBM unavailable: {error}"
        logger.info("Model Trainer Initialized")

    @staticmethod
    def _parameters(estimator):
        if estimator is None or not hasattr(estimator, "get_params"):
            return "{}"
        return str(estimator.get_params(deep=False))

    @staticmethod
    def _score_for_auc(model, features, labels):
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(features)
            classes = list(model.classes_)
            positive_index = classes.index(1) if 1 in classes else -1
            return scores[:, positive_index]
        if hasattr(model, "decision_function"):
            scores = model.decision_function(features)
            classes = list(model.classes_)
            return scores if classes[-1] == 1 else -scores
        return None

    def initiate_model_training(
        self, X_train, X_validation, y_train, y_validation
    ) -> ModelTrainerArtifacts:
        try:
            logger.info("Candidate model training started")
            trained_models = {}
            results = []

            for model_name, estimator in self.models.items():
                started = time.perf_counter()
                try:
                    if estimator is None:
                        raise ImportError(
                            self.dependency_errors.get(
                                model_name, "Candidate estimator is unavailable"
                            )
                        )
                    estimator.fit(X_train, y_train)
                    model = estimator
                    if not hasattr(model, "predict_proba"):
                        model = CalibratedClassifierCV(
                            estimator=estimator, method="sigmoid", cv=3
                        )
                        model.fit(X_train, y_train)
                    training_seconds = time.perf_counter() - started

                    evaluation_started = time.perf_counter()
                    predictions = model.predict(X_validation)
                    scores = self._score_for_auc(model, X_validation, y_validation)
                    try:
                        roc_auc = (
                            float(roc_auc_score(y_validation, scores))
                            if scores is not None
                            else np.nan
                        )
                    except (ValueError, TypeError):
                        roc_auc = np.nan
                    evaluation_seconds = time.perf_counter() - evaluation_started
                    metrics = {
                        "Accuracy": float(accuracy_score(y_validation, predictions)),
                        "Precision": float(
                            precision_score(y_validation, predictions, zero_division=0)
                        ),
                        "Recall": float(
                            recall_score(y_validation, predictions, zero_division=0)
                        ),
                        "F1 Score": float(
                            f1_score(y_validation, predictions, zero_division=0)
                        ),
                        "ROC AUC": roc_auc,
                    }
                    trained_models[model_name] = model
                    results.append(
                        {
                            "Model": model_name,
                            "algorithm": model_name,
                            "parameters": self._parameters(estimator),
                            "training_status": "success",
                            "error_message": "",
                            "validation_accuracy": metrics["Accuracy"],
                            "validation_precision": metrics["Precision"],
                            "validation_recall": metrics["Recall"],
                            "validation_f1": metrics["F1 Score"],
                            "validation_roc_auc": roc_auc,
                            **metrics,
                            "Confusion Matrix": confusion_matrix(
                                y_validation, predictions
                            ).tolist(),
                            "Training Time": training_seconds,
                            "Evaluation Time": evaluation_seconds,
                            "Status": "success",
                            "Error": "",
                        }
                    )
                    logger.info(
                        "%s validation F1=%.4f", model_name, metrics["F1 Score"]
                    )
                except Exception as error:
                    results.append(
                        {
                            "Model": model_name,
                            "algorithm": model_name,
                            "parameters": self._parameters(estimator),
                            "training_status": "failed",
                            "error_message": f"{type(error).__name__}: {error}",
                            "validation_accuracy": np.nan,
                            "validation_precision": np.nan,
                            "validation_recall": np.nan,
                            "validation_f1": np.nan,
                            "validation_roc_auc": np.nan,
                            "Accuracy": np.nan,
                            "Precision": np.nan,
                            "Recall": np.nan,
                            "F1 Score": np.nan,
                            "ROC AUC": np.nan,
                            "Confusion Matrix": None,
                            "Training Time": time.perf_counter() - started,
                            "Evaluation Time": np.nan,
                            "Status": "failed",
                            "Error": f"{type(error).__name__}: {error}",
                        }
                    )
                    logger.exception("Candidate model failed: %s", model_name)

            comparison_df = pd.DataFrame(results)
            eligible = comparison_df[
                (comparison_df["Status"] == "success")
                & np.isfinite(comparison_df["F1 Score"])
            ].copy()
            if eligible.empty:
                raise RuntimeError("No candidate model produced a valid validation F1.")
            eligible["_auc_rank"] = eligible["ROC AUC"].fillna(-np.inf)
            eligible.sort_values(
                ["F1 Score", "_auc_rank"], ascending=False, inplace=True
            )
            selected_name = eligible.iloc[0]["Model"]
            selected_row = eligible.iloc[0]
            comparison_df["Selected"] = comparison_df["Model"] == selected_name
            comparison_df.sort_values(
                ["Selected", "F1 Score"],
                ascending=[False, False],
                na_position="last",
                inplace=True,
            )
            comparison_df.reset_index(drop=True, inplace=True)
            os.makedirs(os.path.dirname(self.config.comparison_path), exist_ok=True)
            comparison_df.to_csv(self.config.comparison_path, index=False)
            logger.info("Model comparison saved to %s", self.config.comparison_path)
            logger.info(
                "Selected %s by validation F1=%.4f (ROC-AUC tie-break)",
                selected_name,
                selected_row["F1 Score"],
            )
            return ModelTrainerArtifacts(
                trained_model=trained_models[selected_name],
                best_model_name=selected_name,
                comparison_dataframe=comparison_df,
                best_parameters={},
                trained_models=trained_models,
                validation_metrics={
                    key: float(selected_row[key])
                    for key in (
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1 Score",
                        "ROC AUC",
                    )
                },
            )
        except Exception as error:
            logger.exception("Model Training Failed")
            raise CustomException(error, __import__("sys"))
