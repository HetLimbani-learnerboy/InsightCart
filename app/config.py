"""
Project : InsightCart

File : config.py

Purpose :
Store all FastAPI configuration values using pydantic-settings.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    PROJECT_NAME: str = "InsightCart"
    PROJECT_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DESCRIPTION: str = "AI Powered Review Detection API"

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    DATABASE_URL: str = "sqlite:///./test.db"

    HF_SENTIMENT_ENABLED: bool = True
    HF_TOKEN: str = ""
    HF_SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    HF_SENTIMENT_TIMEOUT: float = Field(default=30.0, gt=0)
    HF_SENTIMENT_MAX_LENGTH: int = Field(default=512, ge=1, le=5000)

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TIMEOUT: float = Field(default=30.0, gt=0, le=300)
    REVIEW_SUMMARY_MAX_REVIEWS: int = Field(default=1000, ge=1, le=10000)
    REVIEW_SUMMARY_BATCH_TOKENS: int = Field(default=6000, ge=256, le=20000)
    REVIEW_SUMMARY_CACHE_TTL_HOURS: int = Field(default=24, ge=1, le=8760)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
