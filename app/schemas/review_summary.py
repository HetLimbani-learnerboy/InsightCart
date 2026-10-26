"""Contracts for product review summarization."""

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator


class ReviewInput(BaseModel):
    review_id: Optional[str] = Field(default=None, max_length=255)
    title: Optional[str] = Field(default=None, max_length=1000)
    text: str = Field(
        min_length=1,
        max_length=20000,
        validation_alias=AliasChoices("text", "review"),
    )
    rating: Optional[float] = Field(default=None, ge=1, le=5)
    review_date: Optional[date] = None
    verified_purchase: Optional[bool] = None
    source: Optional[str] = Field(default=None, max_length=100)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Review text must not be empty")
        return value


class SummarizeReviewsRequest(BaseModel):
    product_id: Optional[str] = Field(default=None, max_length=255)
    product_name: str = Field(min_length=1, max_length=500)
    product_url: Optional[str] = Field(default=None, max_length=2000)
    reviews: list[ReviewInput] = Field(min_length=1, max_length=1000)

    @field_validator("product_name")
    @classmethod
    def trim_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Product name must not be empty")
        return value.strip()


class Finding(BaseModel):
    topic: str = Field(min_length=1, max_length=160)
    finding: str = Field(min_length=1, max_length=1200)
    evidence_count: int = Field(ge=0)
    evidence_review_ids: list[str]


class KeyTopic(BaseModel):
    topic: str
    finding: str
    sentiment: Literal["positive", "neutral", "negative", "mixed"]


class DateRange(BaseModel):
    start: Optional[date] = None
    end: Optional[date] = None


class SentimentStats(BaseModel):
    analyzed: int = Field(ge=0)
    failed: int = Field(ge=0)
    positive: int = Field(ge=0)
    neutral: int = Field(ge=0)
    negative: int = Field(ge=0)
    percentages: dict[str, float]


class ProductReviewSummaryResponse(BaseModel):
    product_id: Optional[str]
    product_name: str
    overall_summary: str
    pros: list[Finding]
    cons: list[Finding]
    key_topics: list[KeyTopic]
    recommendation: str
    reviews_analyzed: int
    review_date_range: DateRange
    generated_at: datetime
    model: str
    sentiment: SentimentStats
    ai_generated: bool = True
