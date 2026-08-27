"""
Project : InsightCart

File : connection.py

Purpose :
PostgreSQL database connection using SQLAlchemy.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings


DATABASE_URL = settings.DATABASE_URL


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


Base = declarative_base()


def get_db():
    """
    Provide database session for FastAPI endpoints.
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()