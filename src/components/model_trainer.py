"""
Project : InsightCart

File : model_trainer.py

Purpose :
Train baseline machine learning models, perform hyperparameter tuning,
and calibrate the top model.
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
                random_state=self.config.random_state, max_iter=1000
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
                random_state=self.config.random_state, max_iter=10000,dual="auto"
            ),
            "XGBoost": XGBClassifier(
                random_state=self.config.random_state,
                eval_metric="logloss",
            ),
            "LightGBM": LGBMClassifier(
                random_state=self.config.random_state, verbose=-1
            ),
        }
        logger.info("Model Trainer Initialized")

    def initiate_model_training(
        self, X_train, X_test, y_train, y_test
    ) -> ModelTrainerArtifacts:
        try:
            logger.info("Model Training Started")

            # --- 1. Evaluate Baseline Models ---
            model_results = []
            trained_models = {}

            logger.info(f"Total Models to Evaluate: {len(self.models)}")

            for model_name, model in self.models.items():
                logger.info(f"Training Baseline: {model_name}")

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

                if model_name == "Linear SVM":
                    decision_scores = model.decision_function(X_test)
                    roc_auc = roc_auc_score(y_test, decision_scores)
                else:
                    probabilities = model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, probabilities)

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

            comparison_df = (
                pd.DataFrame(model_results)
                .sort_values(by="F1 Score", ascending=False)
                .reset_index(drop=True)
            )

            logger.info("\nBaseline Models Evaluation:\n%s", comparison_df)

            best_baseline_name = comparison_df.iloc[0]["Model"]
            logger.info(f"Best Baseline Model Selected: {best_baseline_name}")

            # --- 2. Hyperparameter Tuning ---
            logger.info("Starting Hyperparameter Tuning for Linear SVM")

            svm_parameter_grid = {
                "C": [0.01, 0.1, 1, 10],
                "loss": ["hinge", "squared_hinge"],
            }

            base_svm = LinearSVC(
                random_state=self.config.random_state, max_iter=5000
            )

            grid_search = GridSearchCV(
                estimator=base_svm,
                param_grid=svm_parameter_grid,
                scoring="f1",
                cv=5,
                n_jobs=-1,
                verbose=0,
            )

            grid_search.fit(X_train, y_train)
            logger.info("Grid Search Completed")
            logger.info(f"Best Hyperparameters: {grid_search.best_params_}")

            best_svm = grid_search.best_estimator_

            # --- 3. Model Calibration ---
            logger.info("Starting Model Calibration")

            calibrated_svm = CalibratedClassifierCV(
                estimator=best_svm, method="sigmoid", cv=5
            )

            calibrated_svm.fit(X_train, y_train)
            logger.info("Model Calibration Completed")

            # --- 4. Return Trainer Artifacts ---
            trainer_artifacts = ModelTrainerArtifacts(
                trained_model=calibrated_svm,
                best_model_name="Calibrated Linear SVM",
                comparison_dataframe=comparison_df,
                best_parameters=grid_search.best_params_,
            )

            logger.info("Model Training Completed Successfully")
            return trainer_artifacts

        except Exception as e:
            logger.error("Model Training Failed")
            raise CustomException(e, sys)