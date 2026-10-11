"""
Project : InsightCart

File : review_detection_training_pipeline.py

Purpose :
Execute the end-to-end ML training, evaluation, quality gating, and pushing pipeline.
"""

import json
import os
import shutil
import sys
import tempfile
import warnings

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from src.components.data_validation import DataValidation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.components.model_trainer import ModelTrainer
from src.evaluation.model_gate import model_quality_gate
from src.logger import logger

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TrainingPipeline:

    def run_pipeline(self):
        logger.info("Training Pipeline Started")

        ingestion = DataIngestion()
        dataframe = ingestion.initiate_data_ingestion()

        validation = DataValidation()
        dataframe = validation.validate_dataset(dataframe)

        staging_dir = tempfile.mkdtemp(prefix="insightcart-preprocessing-")
        try:
            transformation = DataTransformation(
                DataTransformationConfig(
                    tfidf_path=os.path.join(staging_dir, "tfidf_vectorizer.pkl"),
                    encoder_path=os.path.join(staging_dir, "category_encoder.pkl"),
                    scaler_path=os.path.join(staging_dir, "rating_scaler.pkl"),
                )
            )
            transformation_artifacts = transformation.initiate_data_transformation(
                dataframe
            )
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

        trainer = ModelTrainer()
        trainer_artifacts = trainer.initiate_model_training(
            transformation_artifacts.X_train,
            transformation_artifacts.X_validation,
            transformation_artifacts.y_train,
            transformation_artifacts.y_validation,
        )

        evaluator = ModelEvaluation()
        evaluator.initiate_model_evaluation(
            trainer_artifacts,
            transformation_artifacts.X_test,
            transformation_artifacts.y_test,
        )

        eval_metrics_path = "artifacts/evaluation_results.json"
        new_metrics = None
        try:
            with open(eval_metrics_path, "r", encoding="utf-8") as metrics_file:
                new_metrics = json.load(metrics_file)
        except (OSError, json.JSONDecodeError) as error:
            logger.error("Evaluation metrics unavailable; promotion blocked: %s", error)

        prod_metrics_path = "artifacts/production_metrics.json"
        prod_metrics = None
        try:
            with open(prod_metrics_path, "r", encoding="utf-8") as metrics_file:
                prod_metrics = json.load(metrics_file)
        except (OSError, json.JSONDecodeError) as error:
            logger.error(
                "Production baseline unavailable; promotion blocked: %s", error
            )

        if new_metrics is None:
            logger.error("Evaluation metrics not found. Model promotion blocked.")
            passed = False
        else:
            passed = model_quality_gate(
                new_metrics,
                prod_metrics,
                minimum_f1=0.90,
            )

        pusher = ModelPusher()
        if passed:
            model_path = pusher.initiate_model_pusher(
                trainer_artifacts,
                evaluation_metrics=new_metrics,
                transformation_artifacts=transformation_artifacts,
            )
            logger.info(f"Model promoted to production: {model_path}")
        else:
            logger.info("Model did not pass quality gate. Not promoting to production.")
            model_path = None

        logger.info("Pipeline Completed Successfully")
        logger.info(f"Model Saved At : {model_path}")


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
