"""
Project : InsightCart

File : model_trainer.py

Purpose :
Train multiple machine learning models, compare their performance,
automatically select the best model, tune it, and calibrate it
when required.
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logger


@dataclass
class ModelTrainerConfig:
    random_state: int = 42
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
    comparison_path: str = "data/processed/model_comparison.csv"


@dataclass
class ModelTrainerArtifacts:
    trained_model: Any
    best_model_name: str
    comparison_dataframe: pd.DataFrame
    best_parameters: Dict[str, Any]


class ModelTrainer:

    def __init__(self):
        self.config = ModelTrainerConfig()
        self.models = {
            "Logistic Regression": LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000,
            ),
            "Multinomial Naive Bayes": MultinomialNB(),
            "Decision Tree": DecisionTreeClassifier(
                random_state=self.config.random_state
            ),
            "Random Forest": RandomForestClassifier(
                random_state=self.config.random_state,
                n_estimators=200,
                n_jobs=-1,
            ),
            "Linear SVM": LinearSVC(
                random_state=self.config.random_state,
                max_iter=10000,
                dual="auto",
            ),
            "XGBoost": XGBClassifier(
                random_state=self.config.random_state,
                eval_metric="logloss",
            ),
            "LightGBM": LGBMClassifier(
                random_state=self.config.random_state,
                verbose=-1,
            ),
        }
        logger.info("Model Trainer Initialized")

    def initiate_model_training(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
    ) -> ModelTrainerArtifacts:
        try:
            logger.info("Model Training Started")
            model_results = []
            trained_models = {}

            logger.info(f"Total Models: {len(self.models)}")

            for model_name, model in self.models.items():
                logger.info(f"Training Model: {model_name}")
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                trained_models[model_name] = model

                prediction_start = time.time()
                predictions = model.predict(X_test)
                prediction_time = time.time() - prediction_start

                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, zero_division=0)
                recall = recall_score(y_test, predictions, zero_division=0)
                f1 = f1_score(y_test, predictions, zero_division=0)

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, probabilities)
                elif hasattr(model, "decision_function"):
                    decision_scores = model.decision_function(X_test)
                    roc_auc = roc_auc_score(y_test, decision_scores)
                else:
                    roc_auc = 0.0

                model_results.append(
                    {
                        "Model": model_name,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1 Score": f1,
                        "ROC AUC": roc_auc,
                        "Training Time": training_time,
                        "Prediction Time": prediction_time,
                    }
                )

                logger.info(
                    f"{model_name} | Accuracy={accuracy:.4f} | F1={f1:.4f}"
                )

            comparison_df = (
                pd.DataFrame(model_results)
                .sort_values(by="F1 Score", ascending=False)
                .reset_index(drop=True)
            )

            logger.info("\nModel Comparison:\n%s", comparison_df)
            comparison_df.to_csv(self.config.comparison_path, index=False)
            logger.info(f"Model comparison saved to: {self.config.comparison_path}")

            best_model_name = comparison_df.iloc[0]["Model"]
            best_baseline_model = trained_models[best_model_name]
            best_baseline_f1 = comparison_df.iloc[0]["F1 Score"]

            logger.info(f"Best Baseline Model: {best_model_name}")
            logger.info(f"Best Baseline F1: {best_baseline_f1:.4f}")

            tuned_model = best_baseline_model
            best_parameters = {}

            logger.info(f"Starting Hyperparameter Tuning for: {best_model_name}")

            if best_model_name == "Logistic Regression":
                parameter_grid = {"C": [0.01, 0.1, 1, 10]}
                grid = GridSearchCV(
                    LogisticRegression(max_iter=1000),
                    parameter_grid,
                    scoring="f1",
                    cv=5,
                    n_jobs=-1,
                )
                grid.fit(X_train, y_train)
                tuned_model = grid.best_estimator_
                best_parameters = grid.best_params_

            elif best_model_name == "Random Forest":
                parameter_grid = {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                }
                grid = GridSearchCV(
                    RandomForestClassifier(random_state=42, n_jobs=-1),
                    parameter_grid,
                    scoring="f1",
                    cv=5,
                    n_jobs=-1,
                )
                grid.fit(X_train, y_train)
                tuned_model = grid.best_estimator_
                best_parameters = grid.best_params_

            elif best_model_name == "Decision Tree":
                parameter_grid = {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                }
                grid = GridSearchCV(
                    DecisionTreeClassifier(random_state=42),
                    parameter_grid,
                    scoring="f1",
                    cv=5,
                    n_jobs=-1,
                )
                grid.fit(X_train, y_train)
                tuned_model = grid.best_estimator_
                best_parameters = grid.best_params_

            elif best_model_name == "Linear SVM":
                parameter_grid = {
                    "C": [0.01, 0.1, 1, 10],
                    "loss": ["hinge", "squared_hinge"],
                }
                grid = GridSearchCV(
                    LinearSVC(random_state=42, max_iter=10000),
                    parameter_grid,
                    scoring="f1",
                    cv=5,
                    n_jobs=-1,
                )
                grid.fit(X_train, y_train)
                tuned_model = grid.best_estimator_
                best_parameters = grid.best_params_

            elif best_model_name == "XGBoost":
                parameter_grid = {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6],
                    "learning_rate": [0.05, 0.1],
                }
                grid = GridSearchCV(
                    XGBClassifier(random_state=42, eval_metric="logloss"),
                    parameter_grid,
                    scoring="f1",
                    cv=5,
                    n_jobs=-1,
                )
                grid.fit(X_train, y_train)
                tuned_model = grid.best_estimator_
                best_parameters = grid.best_params_

            elif best_model_name == "LightGBM":
                parameter_grid = {
                    "n_estimators": [100, 200],
                    "num_leaves": [15, 31],
                    "learning_rate": [0.05, 0.1],
                }
                grid = GridSearchCV(
                    LGBMClassifier(random_state=42, verbose=-1),
                    parameter_grid,
                    scoring="f1",
                    cv=5,
                    n_jobs=-1,
                )
                grid.fit(X_train, y_train)
                tuned_model = grid.best_estimator_
                best_parameters = grid.best_params_

            elif best_model_name == "Multinomial Naive Bayes":
                parameter_grid = {"alpha": [0.1, 0.5, 1.0]}
                grid = GridSearchCV(
                    MultinomialNB(),
                    parameter_grid,
                    scoring="f1",
                    cv=5,
                    n_jobs=-1,
                )
                grid.fit(X_train, y_train)
                tuned_model = grid.best_estimator_
                best_parameters = grid.best_params_

            logger.info(f"Best Parameters: {best_parameters}")

            # Compare tuned model to baseline and consider calibration before selecting final model
            try:
                # Baseline metrics
                baseline_model = best_baseline_model
                baseline_preds = baseline_model.predict(X_test)
                baseline_f1 = f1_score(y_test, baseline_preds, zero_division=0)

                baseline_calib_f1 = baseline_f1
                baseline_final_model = baseline_model
                if not hasattr(baseline_model, "predict_proba"):
                    logger.info("Calibrating baseline model for fair comparison...")
                    baseline_calibrated = CalibratedClassifierCV(
                        estimator=baseline_model, method="sigmoid", cv=5
                    )
                    baseline_calibrated.fit(X_train, y_train)
                    baseline_calib_preds = baseline_calibrated.predict(X_test)
                    baseline_calib_f1 = f1_score(y_test, baseline_calib_preds, zero_division=0)
                    baseline_final_model = baseline_calibrated

                # Tuned model metrics
                tuned_preds = tuned_model.predict(X_test)
                tuned_f1 = f1_score(y_test, tuned_preds, zero_division=0)

                tuned_calib_f1 = tuned_f1
                tuned_final_model = tuned_model
                tuned_was_calibrated = False
                if not hasattr(tuned_model, "predict_proba"):
                    logger.info("Calibrating tuned model for fair comparison...")
                    tuned_calibrated = CalibratedClassifierCV(
                        estimator=tuned_model, method="sigmoid", cv=5
                    )
                    tuned_calibrated.fit(X_train, y_train)
                    tuned_calib_preds = tuned_calibrated.predict(X_test)
                    tuned_calib_f1 = f1_score(y_test, tuned_calib_preds, zero_division=0)
                    tuned_final_model = tuned_calibrated
                    tuned_was_calibrated = True

                logger.info(
                    f"Baseline F1: {baseline_f1:.4f} | Baseline calibrated F1: {baseline_calib_f1:.4f}"
                )
                logger.info(
                    f"Tuned F1: {tuned_f1:.4f} | Tuned calibrated F1: {tuned_calib_f1:.4f}"
                )

                # Select final model based on calibrated F1
                if tuned_calib_f1 > baseline_calib_f1:
                    final_model = tuned_final_model
                    final_model_name = best_model_name
                    if tuned_was_calibrated:
                        final_model_name = f"Calibrated {best_model_name}"
                    logger.info("Tuned model outperformed baseline after calibration — selecting tuned model.")
                else:
                    final_model = baseline_final_model
                    final_model_name = best_model_name
                    best_parameters = {}
                    logger.info("Tuned model did not outperform baseline after calibration — keeping baseline model.")

                final_predictions = final_model.predict(X_test)
                final_f1 = f1_score(y_test, final_predictions, zero_division=0)

                logger.info(f"FINAL MODEL: {final_model_name}")
                logger.info(f"FINAL F1: {final_f1:.4f}")

            except Exception as e:
                logger.error("Error during tuned vs baseline comparison, falling back to tuned model")
                final_model = tuned_model
                final_model_name = best_model_name

            return ModelTrainerArtifacts(
                trained_model=final_model,
                best_model_name=final_model_name,
                comparison_dataframe=comparison_df,
                best_parameters=best_parameters,
            )

        except Exception as e:
            logger.error("Model Training Failed")
            raise CustomException(e, sys)