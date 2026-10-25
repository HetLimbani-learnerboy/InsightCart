"""Sentiment analysis endpoints, separate from review authenticity."""

import logging
import time
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import settings
from app.metrics import (
    SENTIMENT_REQUEST_COUNT,
    SENTIMENT_REQUEST_LATENCY,
)
from app.schemas.sentiment import (
    SentimentBatchRequest,
    SentimentBatchResponse,
    SentimentRequest,
    SentimentResponse,
)
from app.services.sentiment_service import (
    SentimentService,
    SentimentServiceError,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/sentiment",
    tags=["Sentiment Analysis"],
)


@lru_cache(maxsize=1)
def get_sentiment_service() -> SentimentService:
    """
    Reuse one service instance across requests.

    This allows the tokenizer and model to remain cached
    instead of creating a new service for every request.
    """
    return SentimentService(config=settings)


def _raise_sentiment_error(error: SentimentServiceError) -> None:
    """Convert a controlled service error into an HTTP response."""

    raise HTTPException(
        status_code=error.status_code,
        detail={
            "code": error.code,
            "message": error.public_message,
        },
    )


@router.post(
    "/analyze",
    response_model=SentimentResponse,
)
async def analyze_sentiment(
    request: SentimentRequest,
    service: SentimentService = Depends(get_sentiment_service),
):
    """Analyze the sentiment of a single review."""
    review = request.review.strip()

    if len(review) > settings.HF_SENTIMENT_MAX_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "review_too_long",
                "message": (
                    "Review exceeds the "
                    f"{settings.HF_SENTIMENT_MAX_LENGTH}-character "
                    "sentiment limit."
                ),
            },
        )

    started = time.perf_counter()

    try:
        result = await service.analyze(review)

    except SentimentServiceError as error:
        SENTIMENT_REQUEST_COUNT.labels(outcome="error").inc()

        logger.warning(
            "Single sentiment analysis failed: %s",
            error.code,
        )

        _raise_sentiment_error(error)

    except Exception as error:
        SENTIMENT_REQUEST_COUNT.labels(outcome="error").inc()

        logger.exception("Unexpected error during single sentiment analysis")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "sentiment_internal_error",
                "message": "Sentiment analysis failed unexpectedly.",
            },
        ) from error

    duration = time.perf_counter() - started

    SENTIMENT_REQUEST_COUNT.labels(outcome="success").inc()

    SENTIMENT_REQUEST_LATENCY.observe(duration)

    return {
        "success": True,
        "sentiment": result,
        "model": {
            "provider": "huggingface",
            "name": settings.HF_SENTIMENT_MODEL,
        },
        "processing_time_ms": round(duration * 1000, 2),
    }


@router.post(
    "/analyze/batch",
    response_model=SentimentBatchResponse,
)
async def analyze_sentiment_batch(
    request: SentimentBatchRequest,
    service: SentimentService = Depends(get_sentiment_service),
):
    """Analyze multiple reviews and return individual predictions."""

    started = time.perf_counter()

    try:
        results = await service.analyze_batch(request.reviews)

    except SentimentServiceError as error:
        # Handle service-level errors that occur outside
        # individual review processing.
        logger.exception(
            "Batch sentiment analysis failed: %s",
            error.code,
        )

        _raise_sentiment_error(error)

    except Exception as error:
        logger.exception("Unexpected error during batch sentiment analysis")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "sentiment_batch_internal_error",
                "message": ("Batch sentiment analysis failed unexpectedly."),
            },
        ) from error

    elapsed = time.perf_counter() - started
    review_count = len(results)

    # Update request metrics and attach model metadata.
    for result in results:
        available = result.get("available", False)

        SENTIMENT_REQUEST_COUNT.labels(
            outcome="success" if available else "error"
        ).inc()

        result["model"] = (
            {
                "provider": "huggingface",
                "name": settings.HF_SENTIMENT_MODEL,
            }
            if available
            else None
        )

    # Record average latency per review.
    if review_count > 0:
        average_latency = elapsed / review_count

        for _ in results:
            SENTIMENT_REQUEST_LATENCY.observe(average_latency)

    logger.info(
        "Batch sentiment analysis completed: "
        "total_reviews=%d successful=%d failed=%d elapsed_ms=%.2f",
        review_count,
        sum(1 for result in results if result.get("available")),
        sum(1 for result in results if not result.get("available")),
        elapsed * 1000,
    )

    return {
        "total_reviews": review_count,
        "results": results,
    }
