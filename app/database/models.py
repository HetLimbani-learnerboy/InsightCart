"""
Project : InsightCart

File : models.py

Purpose :
SQLAlchemy database models.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
)

from app.database.connection import Base


class ReviewPrediction(Base):

    __tablename__ = "review_predictions"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    title = Column(
        String(500),
        nullable=True,
    )

    review = Column(
        Text,
        nullable=False,
    )

    clean_review = Column(
        Text,
        nullable=True,
    )

    rating = Column(
        Integer,
        nullable=False,
    )

    category = Column(
        String(255),
        nullable=False,
    )

    prediction = Column(
        Integer,
        nullable=False,
    )

    review_type = Column(
        String(100),
        nullable=False,
    )

    confidence = Column(
        String(50),
        nullable=False,
    )

    reason = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )