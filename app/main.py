"""
Project : InsightCart

File : main.py

Purpose :
Application Entry Point.
"""

import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from prometheus_client import generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

from app.config import settings
from app.database.init_db import init_db
from app.tracing import configure_tracing

from app.routes.health import router as health_router
from app.routes.review_detection import router as review_router
from app.routes.sentiment import router as sentiment_router

from app.middleware.prometheus import PrometheusMiddleware


@asynccontextmanager
async def lifespan(_app):
    # Review prediction endpoints persist results, so ensure the database schema
    # exists before the app starts accepting requests.
    init_db()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure once before request handling; tracing setup is best effort.
configure_tracing()

app.add_middleware(PrometheusMiddleware)

# Enable CORS for external frontend or API clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-MLflow-Trace-Id"],
)

# Include Routers
app.include_router(health_router, prefix="")
app.include_router(review_router, prefix=settings.API_PREFIX)
app.include_router(sentiment_router, prefix=settings.API_PREFIX)


@app.get("/", tags=["Home"])
def home():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "status": "Running",
        "documentation": "/docs",
    }


@app.get(
    "/metrics",
    tags=["Monitoring"],
)
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
