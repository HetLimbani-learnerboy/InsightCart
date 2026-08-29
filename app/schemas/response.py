"""
Project : InsightCart

File : response.py

Purpose :
Response schemas for the Review Detection API.
"""

from typing import List
from pydantic import BaseModel, Field


class ReviewPredictionResponse(BaseModel):

    title: str = Field(
        default="Untitled Review",
        description="Review title",
        examples=["Absolutely great if you have pets"],
    )

    prediction: int = Field(
        ...,
        description="Prediction code (0 = Human-Written, 1 = AI-Generated)",
        examples=[0],
    )

    review_type: str = Field(
        ...,
        description="Human-readable classification label",
        examples=["Human-Written Review"],
    )

    confidence: str = Field(
        ...,
        description="Qualitative confidence indicator (very_high, high, moderate, low)",
        examples=["very_high"],
    )

    category: str = Field(
        ...,
        description="Amazon product category",
        examples=["Electronics_5"],
    )

    rating: int = Field(
        ...,
        description="Product rating",
        examples=[5],
    )

    clean_review: str = Field(
        ...,
        description="Cleaned/preprocessed review text",
    )

    reason: str = Field(
        ...,
        description="Explanation for the classification result",
        examples=["The writing style resembles authentic human-written reviews."],
    )


class BatchPredictionItem(BaseModel):

    title: str = Field(
        default="Untitled Review",
        description="Review title",
        examples=["Absolutely great if you have pets"],
    )

    prediction: int = Field(
        ...,
        description="Prediction code (0 = Human-Written, 1 = AI-Generated)",
        examples=[0],
    )

    review_type: str = Field(
        ...,
        description="Human-readable classification label",
        examples=["Human-Written Review"],
    )

    confidence: str = Field(
        ...,
        description="Qualitative confidence band (very_high, high, moderate, low)",
        examples=["very_high"],
    )

    category: str = Field(
        ...,
        description="Amazon product category",
        examples=["Electronics_5"],
    )

    rating: int = Field(
        ...,
        description="Product rating",
        examples=[5],
    )

    clean_review: str = Field(
        ...,
        description="Cleaned/preprocessed review text",
    )

    reason: str = Field(
        ...,
        description="Explanation for the classification result",
    )


class BatchPredictionResponse(BaseModel):

    total_reviews: int = Field(
        ...,
        description="Total number of reviews processed",
        examples=[1],
    )

    results: List[BatchPredictionItem] = Field(
        ...,
        description="List of prediction results for each input review",
    )


class HealthResponse(BaseModel):

    status: str = Field(..., examples=["healthy"])
    project: str = Field(..., examples=["InsightCart"])
    version: str = Field(..., examples=["1.0.0"])
