"""
Project : InsightCart

File : request.py

Purpose :
Request schemas for the Review Detection API.
"""

from typing import List
from pydantic import BaseModel, Field


class ReviewPredictionRequest(BaseModel):

    review: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Customer review text",
        examples=["Excellent battery backup and premium build quality."],
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

    review: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Customer review text",
        examples=["Great product, works as described."],
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


class BatchPredictionRequest(BaseModel):

    reviews: List[BatchReviewItem] = Field(
        ...,
        min_length=1,
        description="List of customer reviews to analyze",
    )