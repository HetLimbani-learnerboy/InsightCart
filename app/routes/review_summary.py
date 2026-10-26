"""Product review summarization endpoints."""

import logging
import uuid
from importlib.util import find_spec
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database.connection import get_db
from app.database.models import ProductReviewSummary
from app.schemas.review_summary import ProductReviewSummaryResponse, SummarizeReviewsRequest
from app.services.review_provider import UnconfiguredReviewProvider
from app.services.review_summary_service import ReviewSummaryService, SummaryServiceError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/review-summary", tags=["Review Summarization"])


@lru_cache(maxsize=1)
def get_summary_service() -> ReviewSummaryService:
    return ReviewSummaryService()


@router.post("/summarize", response_model=ProductReviewSummaryResponse)
async def summarize_reviews(
    request: SummarizeReviewsRequest,
    db: Session = Depends(get_db),
    service: ReviewSummaryService = Depends(get_summary_service),
):
    try:
        return await service.summarize(request, db)
    except SummaryServiceError as error:
        logger.warning("Review summary request failed: code=%s", error.code)
        raise HTTPException(status_code=error.status_code, detail={"code": error.code, "message": error.message}) from error
    except Exception as error:
        request_id = uuid.uuid4().hex[:12]
        try:
            db.rollback()
        except Exception as rollback_error:
            logger.error(
                "Review summary rollback failed request_id=%s error_type=%s",
                request_id,
                type(rollback_error).__name__,
            )
        logger.exception(
            "Review summarization failed request_id=%s error_type=%s",
            request_id,
            type(error).__name__,
        )
        raise HTTPException(status_code=500, detail={
            "code": "summary_internal_error",
            "message": "Review summarization failed. Check the backend log using the reference ID.",
            "request_id": request_id,
        }) from error


@router.post("/products/{product_id}/analyze", response_model=ProductReviewSummaryResponse)
async def collect_and_summarize(product_id: str, db: Session = Depends(get_db)):
    # Provider injection point: only configured, permitted data sources may be used.
    try:
        await UnconfiguredReviewProvider().get_reviews(product_id, settings.REVIEW_SUMMARY_MAX_REVIEWS)
    except RuntimeError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={
            "code": "review_provider_not_configured",
            "message": "No authorized review provider is configured. Supply permitted reviews to POST /review-summary/summarize or configure a provider.",
        }) from error


@router.get("/products/{product_id}", response_model=ProductReviewSummaryResponse)
def latest_summary(product_id: str, db: Session = Depends(get_db)):
    row = db.query(ProductReviewSummary).filter_by(product_id=product_id).order_by(ProductReviewSummary.generated_at.desc()).first()
    if not row:
        raise HTTPException(status_code=404, detail={"code": "summary_not_found", "message": "No summary exists for this product."})
    return row.summary


@router.get("/products/{product_id}/history", response_model=list[ProductReviewSummaryResponse])
def summary_history(product_id: str, db: Session = Depends(get_db)):
    rows = db.query(ProductReviewSummary).filter_by(product_id=product_id).order_by(ProductReviewSummary.generated_at.desc()).limit(100).all()
    return [row.summary for row in rows]


@router.get("/health")
def summary_health():
    sdk_available = find_spec("groq") is not None
    configured = bool(settings.GROQ_API_KEY)
    if not sdk_available:
        state = "sdk_missing"
    elif not configured:
        state = "unconfigured"
    else:
        state = "configured"
    return {
        "status": state,
        "provider": "groq",
        "model": settings.GROQ_MODEL,
        "sdk_available": sdk_available,
        "api_key_configured": configured,
    }
