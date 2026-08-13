"""
Project : InsightCart

File : request.py

Purpose :
Request schemas for the Review Detection API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ReviewPredictionRequest(BaseModel):

    title: Optional[str] = Field(
        default="",
        description="Customer review title",
        examples=["Absolutely great if you have pets"],
    )

    review: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="Customer review text",
        examples=["This one surprised me. I have had a Kirby for 20+ years..."],
    )

    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Product rating (1 to 5)",
        examples=[5],
    )

    category: str = Field(
        ...,
        description="Amazon product category",
        examples=["Electronics_5"],
    )


class BatchReviewItem(BaseModel):

    id: Optional[str] = Field(
        default="",
        description="Unique review identifier",
    )

    title: Optional[str] = Field(
        default="",
        description="Customer review title",
    )

    review: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="Customer review text",
    )

    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Product rating (1 to 5)",
    )

    category: str = Field(
        ...,
        description="Amazon product category",
    )


class BatchPredictionRequest(BaseModel):

    reviews: List[BatchReviewItem] = Field(
        ...,
        min_length=1,
        description="List of customer reviews to analyze",
    )