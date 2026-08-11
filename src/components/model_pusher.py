"""
Project : InsightCart

File : model_pusher.py

Purpose :
Save the trained model into artifacts/models.
"""

import sys
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

    def initiate_model_pusher(self, trainer_artifacts):
        try:
            logger.info("Saving Trained Model")

            save_object(
                self.config.model_path,
                trainer_artifacts.trained_model,
            )

            logger.info(
                f"Model Saved Successfully : {self.config.model_path}"
            )

            return self.config.model_path

        except Exception as e:
            raise CustomException(e, sys)