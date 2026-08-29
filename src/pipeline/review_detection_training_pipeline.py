"""
Project : InsightCart

Training Pipeline
"""

import os
import sys

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import warnings
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.logger import logger
from src.evaluation.model_gate import model_quality_gate
import json
import os


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TrainingPipeline:

    def run_pipeline(self):

        logger.info("Training Pipeline Started")

        ingestion = DataIngestion()
        dataframe = ingestion.initiate_data_ingestion()

        validation = DataValidation()
        dataframe = validation.validate_dataset(dataframe)

        transformation = DataTransformation()
        transformation_artifacts = transformation.initiate_data_transformation(
            dataframe
        )

        trainer = ModelTrainer()
        trainer_artifacts = trainer.initiate_model_training(
            transformation_artifacts.X_train,
            transformation_artifacts.X_test,
            transformation_artifacts.y_train,
            transformation_artifacts.y_test,
        )

        evaluator = ModelEvaluation()
        evaluation_artifacts = evaluator.initiate_model_evaluation(
            trainer_artifacts,
            transformation_artifacts.X_test,
            transformation_artifacts.y_test,
        )
        # Load new evaluation metrics saved by ModelEvaluation (if any)
        eval_metrics_path = "artifacts/evaluation_results.json"
        new_metrics = None
        if os.path.exists(eval_metrics_path):
            with open(eval_metrics_path, "r") as f:
                new_metrics = json.load(f)

        # Load production metrics if they exist
        prod_metrics_path = "artifacts/production_metrics.json"
        prod_metrics = None
        if os.path.exists(prod_metrics_path):
            with open(prod_metrics_path, "r") as f:
                prod_metrics = json.load(f)

        # Run quality gate: if no production metrics exist, allow push
        passed = True
        if new_metrics is None:
            logger.error(
                "Evaluation metrics not found. "
                "Model promotion blocked."
            )
            passed = False
        else:
            passed = model_quality_gate(
                new_metrics,
                prod_metrics or {},
                minimum_f1=0.90,
            )

        pusher = ModelPusher()
        if passed:
            # push model and record production metrics
            model_path = pusher.initiate_model_pusher(trainer_artifacts, evaluation_metrics=new_metrics)
            logger.info(f"Model promoted to production: {model_path}")
        else:
            logger.info("Model did not pass quality gate. Not promoting to production.")
            model_path = None

        logger.info("Pipeline Completed Successfully")
        logger.info(f"Model Saved At : {model_path}")


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()