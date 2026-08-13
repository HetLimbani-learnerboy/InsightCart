"""
Project : InsightCart

File : response.py

Purpose :
Response schemas for the Review Detection API.
"""

from typing import List
from pydantic import BaseModel, Field


class ReviewPredictionResponse(BaseModel):

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

    confidence: float = Field(
        ...,
        description="Prediction confidence percentage",
        examples=[95.50],
    )

    confidence_level: str = Field(
        ...,
        description="Qualitative confidence band (Very High, High, Moderate, Low)",
        examples=["Very High"],
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

    reason: str = Field(
        ...,
        description="Explanation for the classification result",
        examples=["The writing style resembles authentic human-written reviews."],
    )


class BatchPredictionItem(BaseModel):

    prediction: int = Field(
        ...,
        description="Prediction code (0 = Human-Written, 1 = AI-Generated)",
        examples=[1],
    )

    review_type: str = Field(
        ...,
        description="Human-readable classification label",
        examples=["AI-Generated Review"],
    )

    confidence: float = Field(
        ...,
        description="Prediction confidence percentage",
        examples=[98.20],
    )

    confidence_level: str = Field(
        ...,
        description="Qualitative confidence band",
        examples=["Very High"],
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
        examples=["great product work describe fast shipping good quality"],
    )

    reason: str = Field(
        ...,
        description="Explanation for the classification result",
        examples=[
            "The writing style is similar to computer-generated reviews observed during model training."
        ],
    )


class BatchPredictionResponse(BaseModel):

    total_reviews: int = Field(
        ...,
        description="Total number of reviews processed",
        examples=[2],
    )

    results: List[BatchPredictionItem] = Field(
        ...,
        description="List of prediction results for each input review",
    )


class HealthResponse(BaseModel):

    status: str = Field(
        ...,
        description="Health status of the API",
        examples=["healthy"],
    )

    project: str = Field(
        ...,
        description="Project name",
        examples=["InsightCart"],
    )

    version: str = Field(
        ...,
        description="API application version",
        examples=["1.0.0"],
    )