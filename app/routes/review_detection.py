"""
Project : InsightCart

File : review_detection.py

Purpose :
Review Detection API Routes.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.schemas.request import BatchPredictionRequest, ReviewPredictionRequest
from app.schemas.response import BatchPredictionResponse, ReviewPredictionResponse
from app.services.review_service import ReviewDetectionService
from app.tracing import set_span_outputs, traced_span
from src.logger import logger

router = APIRouter(
    prefix="/review",
    tags=["Review Detection"],
)


def get_review_service() -> ReviewDetectionService:
    return ReviewDetectionService()


@router.post(
    "/predict",
    response_model=ReviewPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Single Review",
    description=(
        "Analyze a single customer review to detect if it is human-written or AI-generated."
    ),
)
def predict_review(
    request: ReviewPredictionRequest,
    db: Session = Depends(get_db),
    service: ReviewDetectionService = Depends(get_review_service),
):
    with traced_span(
        "review.predict",
        {
            "endpoint": "/api/v1/review/predict",
            "review_count": 1,
            "review_length": len(request.review),
            "model_version": getattr(
                getattr(service, "pipeline", None), "model_version", "unknown"
            ),
        },
    ) as span:
        try:
            result = service.predict_review(
                review=request.review,
                rating=request.rating,
                category=request.category,
                title=request.title or "",
                db=db,
            )
            with traced_span(
                "review.response_generation",
                {"response_type": "single_prediction"},
            ) as response_span:
                response = ReviewPredictionResponse(
                    title=result["title"],
                    prediction=result["prediction"],
                    review_type=result["review_type"],
                    confidence=result["confidence"],
                    category=result["category"],
                    rating=result["rating"],
                    clean_review=result["clean_review"],
                    reason=result["reason"],
                )
                set_span_outputs(
                    response_span,
                    {"status": "success", "schema": "ReviewPredictionResponse"},
                )
            set_span_outputs(
                span,
                {
                    "status": "success",
                    "prediction": result["prediction"],
                    "review_type": result["review_type"],
                    "confidence": result["confidence"],
                },
            )
            return response
        except Exception as e:
            set_span_outputs(span, {"status": "error", "error_type": type(e).__name__})
            logger.error("Single prediction failed: %s", type(e).__name__)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction failed.",
            )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Multiple Reviews",
    description=(
        "Analyze a batch of reviews where each review specifies its title, category, and rating."
    ),
)
def predict_batch(
    request: BatchPredictionRequest,
    db: Session = Depends(get_db),
    service: ReviewDetectionService = Depends(get_review_service),
):
    with traced_span(
        "review.predict.batch",
        {
            "endpoint": "/api/v1/review/predict/batch",
            "review_count": len(request.reviews),
            "review_lengths": [len(item.review) for item in request.reviews],
            "model_version": getattr(
                getattr(service, "pipeline", None), "model_version", "unknown"
            ),
        },
    ) as span:
        try:
            result = service.predict_batch(reviews=request.reviews, db=db)
            with traced_span(
                "review.response_generation",
                {
                    "response_type": "batch_prediction",
                    "review_count": result["total_reviews"],
                },
            ) as response_span:
                response = BatchPredictionResponse(
                    total_reviews=result["total_reviews"],
                    results=result["results"],
                )
                set_span_outputs(
                    response_span,
                    {
                        "status": "success",
                        "schema": "BatchPredictionResponse",
                        "review_count": result["total_reviews"],
                    },
                )
            set_span_outputs(
                span,
                {
                    "status": "success",
                    "review_count": result["total_reviews"],
                    "prediction_counts": {
                        "ai_generated": sum(
                            item["prediction"] == 1 for item in result["results"]
                        ),
                        "human_written": sum(
                            item["prediction"] == 0 for item in result["results"]
                        ),
                    },
                },
            )
            return response
        except Exception as e:
            set_span_outputs(span, {"status": "error", "error_type": type(e).__name__})
            logger.error("Batch prediction failed: %s", type(e).__name__)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction failed.",
            )
