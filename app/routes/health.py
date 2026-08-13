"""
Project : InsightCart

File : health.py

Purpose :
Health Check API.
"""

from fastapi import APIRouter

from app.config import settings

from app.schemas.response import HealthResponse


router = APIRouter(
    tags=["Health"]
)


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    return HealthResponse(

        status="Healthy",

        project=settings.PROJECT_NAME,

        version=settings.PROJECT_VERSION,

    )