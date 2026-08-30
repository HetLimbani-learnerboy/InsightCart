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
    try:
        result = service.predict_review(
            review=request.review,
            rating=request.rating,
            category=request.category,
            title=request.title or "",
            db=db,
        )
        return ReviewPredictionResponse(
            title=result["title"],
            prediction=result["prediction"],
            review_type=result["review_type"],
            confidence=result["confidence"],
            category=result["category"],
            rating=result["rating"],
            clean_review=result["clean_review"],
            reason=result["reason"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
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
    try:
        result = service.predict_batch(
            reviews=request.reviews,
            db=db,
        )
        return BatchPredictionResponse(
            total_reviews=result["total_reviews"],
            results=result["results"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )