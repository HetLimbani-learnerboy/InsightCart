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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
