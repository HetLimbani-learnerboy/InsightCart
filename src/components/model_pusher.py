"""
Project : InsightCart

File : model_pusher.py

Purpose :
Save the trained model into artifacts/models.
"""

import sys
import os
from dataclasses import dataclass

from src.constants import MODEL_PATH
from src.exception import CustomException
from src.logger import logger
from src.utils.common import save_object


@dataclass
class ModelPusherConfig:
    model_path: str = MODEL_PATH


class ModelPusher:

    def __init__(self):
        self.config = ModelPusherConfig()

    def initiate_model_pusher(self, trainer_artifacts, evaluation_metrics: dict = None):
        try:
            logger.info("Saving Trained Model")

            save_object(
                self.config.model_path,
                trainer_artifacts.trained_model,
            )

            logger.info(
                f"Model Saved Successfully : {self.config.model_path}"
            )

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

            return self.config.model_path

        except Exception as e:
            raise CustomException(e, sys)