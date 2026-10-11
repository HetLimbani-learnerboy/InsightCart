"""
Project : InsightCart

File : model_pusher.py

Purpose :
Save the trained model into artifacts/models.
"""

import sys
import os
import tempfile
from dataclasses import dataclass

import joblib

from src.constants import ENCODER_PATH, MODEL_PATH, SCALER_PATH, TFIDF_PATH
from src.exception import CustomException
from src.logger import logger
from src.utils.common import save_object


@dataclass
class ModelPusherConfig:
    model_path: str = MODEL_PATH
    tfidf_path: str = TFIDF_PATH
    encoder_path: str = ENCODER_PATH
    scaler_path: str = SCALER_PATH


class ModelPusher:

    def __init__(self):
        self.config = ModelPusherConfig()

    def initiate_model_pusher(
        self,
        trainer_artifacts,
        evaluation_metrics: dict = None,
        transformation_artifacts=None,
    ):
        try:
            logger.info("Saving Trained Model")

            bundle_path = None
            if transformation_artifacts is None:
                save_object(self.config.model_path, trainer_artifacts.trained_model)
            else:
                bundle = {
                    "model": trainer_artifacts.trained_model,
                    "tfidf": transformation_artifacts.tfidf,
                    "category_encoder": transformation_artifacts.category_encoder,
                    "rating_scaler": transformation_artifacts.rating_scaler,
                    "model_name": trainer_artifacts.best_model_name,
                }
                bundle_path = os.path.join(
                    os.path.dirname(self.config.model_path),
                    "production_model_bundle.pkl",
                )
                os.makedirs(os.path.dirname(bundle_path), exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=os.path.dirname(bundle_path), delete=False
                ) as temporary_file:
                    temporary_path = temporary_file.name
                try:
                    joblib.dump(bundle, temporary_path)
                    os.replace(temporary_path, bundle_path)
                finally:
                    if os.path.exists(temporary_path):
                        os.remove(temporary_path)
                save_object(self.config.tfidf_path, transformation_artifacts.tfidf)
                save_object(
                    self.config.encoder_path,
                    transformation_artifacts.category_encoder,
                )
                save_object(
                    self.config.scaler_path, transformation_artifacts.rating_scaler
                )
                save_object(self.config.model_path, trainer_artifacts.trained_model)

            logger.info(f"Model Saved Successfully : {self.config.model_path}")

            # Optionally save production metrics when provided
            if evaluation_metrics is not None:
                try:
                    import json

                    prod_metrics_path = "artifacts/production_metrics.json"
                    os.makedirs(os.path.dirname(prod_metrics_path), exist_ok=True)
                    with open(prod_metrics_path, "w") as f:
                        json.dump(evaluation_metrics, f, indent=4)
                    logger.info(f"Production metrics saved to: {prod_metrics_path}")
                except Exception:
                    logger.warning("Failed to save production metrics file")

            return bundle_path or self.config.model_path

        except Exception as e:
            raise CustomException(e, sys)
