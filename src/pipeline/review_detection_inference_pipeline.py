"""
Project : InsightCart

File : review_detection_inference_pipeline.py

Purpose :
Load the trained model and preprocessing artifacts,
prepare the input review, and perform inference with qualitative confidence output.
"""

import os
import sys

# Add project root directory (InsightCart) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataclasses import dataclass
from typing import Any, Dict
import pandas as pd
from scipy.sparse import hstack

from src.constants import (
    CATEGORY_COLUMN,
    ENCODER_PATH,
    MODEL_PATH,
    RATING_COLUMN,
    SCALER_PATH,
    TFIDF_PATH,
)
from src.exception import CustomException
from src.logger import logger
from src.utils.common import load_object
from src.utils.preprocessing import clean_text


@dataclass
class InferenceConfig:
    model_path: str = MODEL_PATH
    tfidf_path: str = TFIDF_PATH
    encoder_path: str = ENCODER_PATH
    scaler_path: str = SCALER_PATH


class ReviewDetectionInferencePipeline:

    def __init__(self):
        try:
            logger.info("Loading Inference Artifacts")

            self.config = InferenceConfig()

            self.model = load_object(self.config.model_path)
            self.vectorizer = load_object(self.config.tfidf_path)
            self.category_encoder = load_object(self.config.encoder_path)
            self.rating_scaler = load_object(self.config.scaler_path)

            logger.info("Inference Artifacts Loaded Successfully")

        except Exception as e:
            logger.error("Failed to load inference artifacts")
            raise CustomException(e, sys)

    def validate_input(self, review: str, rating: int, category: str) -> None:
        if not isinstance(review, str) or len(review.strip()) == 0:
            raise ValueError("Review must be a non-empty string.")

        if rating not in [1, 2, 3, 4, 5]:
            raise ValueError("Rating must be an integer between 1 and 5.")

    def prepare_features(self, cleaned_review: str, rating: int, category: str):
        logger.info("Generating TF-IDF Features")
        review_vector = self.vectorizer.transform([cleaned_review])

        logger.info("Encoding Product Category")
        category_df = pd.DataFrame([[category]], columns=[CATEGORY_COLUMN])
        category_vector = self.category_encoder.transform(category_df)

        logger.info("Scaling Rating Feature")
        rating_df = pd.DataFrame([[rating]], columns=[RATING_COLUMN])
        rating_vector = self.rating_scaler.transform(rating_df)

        final_vector = hstack([review_vector, category_vector, rating_vector])

        return final_vector

    def predict(
        self, review: str, rating: int, category: str, title: str = ""
    ) -> Dict[str, Any]:
        try:
            logger.info("Starting Prediction Process")

            self.validate_input(review, rating, category)

            cleaned_review = clean_text(review)

            feature_vector = self.prepare_features(cleaned_review, rating, category)

            prediction = int(self.model.predict(feature_vector)[0])
            probabilities = self.model.predict_proba(feature_vector)[0]

            confidence = round(float(probabilities[prediction]) * 100, 2)

            if confidence >= 90:
                confidence_level = "very_high"
            elif confidence >= 75:
                confidence_level = "high"
            elif confidence >= 60:
                confidence_level = "moderate"
            else:
                confidence_level = "low"

            if prediction == 1:
                review_type = "AI-Generated Review"
                reason = (
                    "The writing style is similar to computer-generated "
                    "reviews observed during model training."
                )
            else:
                review_type = "Human-Written Review"
                reason = "The writing style resembles authentic human-written reviews."

            response = {
                "title": title or "Untitled Review",
                "prediction": prediction,
                "review_type": review_type,
                "confidence": confidence_level,
                "category": category,
                "rating": rating,
                "clean_review": cleaned_review,
                "reason": reason,
            }

            logger.info("Prediction Completed Successfully")
            return response

        except Exception as e:
            logger.error("Prediction Failed")
            raise CustomException(e, sys)
