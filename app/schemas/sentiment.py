"""Request and response schemas for sentiment analysis."""

from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SentimentRequest(BaseModel):
    review: str = Field(..., min_length=1, max_length=5000)

    @field_validator("review")
    @classmethod
    def trim_review(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Review must not be empty or whitespace-only")
        return value


class SentimentBatchRequest(BaseModel):
    reviews: List[Annotated[str, Field(min_length=1, max_length=5000)]] = Field(
        ..., min_length=1, max_length=40
    )

    @field_validator("reviews")
    @classmethod
    def trim_reviews(cls, reviews: List[str]) -> List[str]:
        trimmed = [review.strip() for review in reviews]
        if any(not review for review in trimmed):
            raise ValueError("Reviews must not be empty or whitespace-only")
        return trimmed


class SentimentResult(BaseModel):
    label: Literal["positive", "neutral", "negative"]
    confidence: float = Field(..., ge=0.0, le=1.0)


class SentimentModelInfo(BaseModel):
    provider: Literal["huggingface"] = "huggingface"
    name: str


class SentimentResponse(BaseModel):
    success: bool = True
    sentiment: SentimentResult
    model: SentimentModelInfo
    processing_time_ms: float = Field(..., ge=0)


class SentimentBatchItem(BaseModel):
    available: bool
    sentiment: Optional[SentimentResult] = None
    model: Optional[SentimentModelInfo] = None
    error: Optional[str] = None


class SentimentBatchResponse(BaseModel):
    total_reviews: int
    results: List[SentimentBatchItem]
