"""
Project : InsightCart

File : review_service.py

Purpose :
Service layer for AI Generated Review Detection.
"""

import sys
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from app.database.models import ReviewPrediction
from app.metrics import PREDICTION_COUNT
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
        title: str = "",
        db: Session = None,
    ) -> Dict[str, Any]:
        try:
            logger.info("Processing Single Review Prediction")

            result = self.pipeline.predict(
                review=review,
                rating=rating,
                category=category,
                title=title,
            )

            PREDICTION_COUNT.labels(
                prediction_type=result["review_type"]
            ).inc()

            if db:
                prediction_record = ReviewPrediction(
                    title=result.get("title") or title,
                    review=review,
                    prediction=result["prediction"],
                    review_type=result["review_type"],
                    confidence=result["confidence"],
                    category=result["category"],
                    rating=result["rating"],
                    clean_review=result["clean_review"],
                    reason=result["reason"],
                )
                db.add(prediction_record)
                db.commit()
                db.refresh(prediction_record)
                logger.info(
                    f"Prediction saved to database with ID {prediction_record.id}"
                )

            logger.info("Prediction Completed Successfully")
            return result

        except Exception as e:
            if db:
                db.rollback()
            logger.error("Single Review Prediction Service Failed")
            raise CustomException(e, sys)

    def predict_batch(
        self,
        reviews: List[Any],
        db: Session = None,
    ) -> Dict[str, Any]:
        try:
            logger.info(f"Batch Prediction Started for {len(reviews)} reviews")
            predictions = []

            for item in reviews:
                title = getattr(item, "title", "") or ""

                prediction = self.pipeline.predict(
                    review=item.review,
                    rating=item.rating,
                    category=item.category,
                    title=title,
                )

                PREDICTION_COUNT.labels(
                    prediction_type=prediction["review_type"]
                ).inc()

                if db:
                    prediction_record = ReviewPrediction(
                        title=prediction.get("title") or title,
                        review=item.review,
                        prediction=prediction["prediction"],
                        review_type=prediction["review_type"],
                        confidence=prediction["confidence"],
                        category=prediction["category"],
                        rating=prediction["rating"],
                        clean_review=prediction["clean_review"],
                        reason=prediction["reason"],
                    )
                    db.add(prediction_record)

                predictions.append(prediction)

            if db:
                db.commit()
                logger.info(
                    f"{len(predictions)} predictions saved to database"
                )

            logger.info("Batch Prediction Completed Successfully")
            return {
                "total_reviews": len(predictions),
                "results": predictions,
            }

        except Exception as e:
            if db:
                db.rollback()
            logger.error("Batch Prediction Failed")
            raise CustomException(e, sys)

    def predict_amazon_reviews(
        self,
        product_name: str,
        reviews: List[Any],
        db: Session = None,
    ) -> Dict[str, Any]:
        try:
            logger.info(f"Processing Amazon Product: {product_name}")
            results = self.predict_batch(
                reviews=reviews,
                db=db,
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