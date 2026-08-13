"""
Project : InsightCart

File : review_service.py

Purpose :
Service layer for AI Generated Review Detection.
"""

import sys
from typing import Any, Dict, List

from src.exception import CustomException
from src.logger import logger
from src.pipeline.review_detection_inference_pipeline import (
    ReviewDetectionInferencePipeline,
)


class ReviewDetectionService:

    def __init__(self):
        logger.info("Initializing Review Detection Service")
        self.pipeline = ReviewDetectionInferencePipeline()

    def predict_review(
        self,
        review: str,
        rating: int,
        category: str,
    ) -> Dict[str, Any]:
        try:
            logger.info("Processing Single Review Prediction")

            result = self.pipeline.predict(
                review=review,
                rating=rating,
                category=category,
            )

            logger.info("Prediction Completed Successfully")
            return result

        except Exception as e:
            logger.error("Single Review Prediction Service Failed")
            raise CustomException(e, sys)

    def predict_batch(
        self,
        reviews: List[Any],
    ) -> Dict[str, Any]:
        try:
            logger.info(f"Batch Prediction Started for {len(reviews)} reviews")

            predictions = []
            for item in reviews:
                prediction = self.pipeline.predict(
                    review=item.review,
                    rating=item.rating,
                    category=item.category,
                )
                predictions.append(prediction)

            logger.info("Batch Prediction Completed Successfully")

            return {
                "total_reviews": len(predictions),
                "results": predictions,
            }

        except Exception as e:
            logger.error("Batch Prediction Failed")
            raise CustomException(e, sys)

    def predict_amazon_reviews(
        self,
        product_name: str,
        reviews: List[Any],
    ) -> Dict[str, Any]:
        try:
            logger.info(f"Processing Amazon Product: {product_name}")

            results = self.predict_batch(
                reviews=reviews,
            )

            return {
                "product_name": product_name,
                **results,
            }

        except Exception as e:
            logger.error(
                f"Amazon Product Review Processing Failed for {product_name}"
            )
            raise CustomException(e, sys)