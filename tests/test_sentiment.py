import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.database.connection import get_db
from app.main import app
from app.routes.review_detection import get_review_service
from app.routes.sentiment import get_sentiment_service
from app.services.sentiment_service import (
    SentimentService,
    SentimentServiceError,
    preprocess_cardiffnlp_text,
)

client = TestClient(app)


class StubSentimentService:
    async def analyze(self, review):
        return {"label": "positive", "confidence": 0.91}

    async def analyze_batch(self, reviews):
        return [
            {
                "available": True,
                "sentiment": {"label": "negative", "confidence": 0.82},
                "error": None,
            },
            {"available": False, "sentiment": None, "error": "review_too_long"},
        ]


@pytest.fixture
def sentiment_override():
    app.dependency_overrides[get_sentiment_service] = StubSentimentService
    yield
    app.dependency_overrides.pop(get_sentiment_service, None)


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, "negative"), (1, "neutral"), (2, "positive")],
)
def test_maps_model_config_labels(index, expected):
    model = SimpleNamespace(
        config=SimpleNamespace(
            id2label={"0": "negative", "1": "neutral", "2": "positive"}
        )
    )
    assert SentimentService._get_label(model, index) == expected


def test_cardiffnlp_preprocessing_replaces_mentions_and_urls():
    assert preprocess_cardiffnlp_text("Thanks @alice https://example.com") == "Thanks @user http"


def test_analyze_uses_worker_thread_and_returns_model_prediction(monkeypatch):
    service = SentimentService(
        Settings(_env_file=None, DEBUG=False, HF_SENTIMENT_ENABLED=True)
    )
    monkeypatch.setattr(
        service,
        "_predict",
        lambda review: {"label": "neutral", "confidence": 0.77},
    )
    result = asyncio.run(service.analyze("A mixed review"))
    assert result == {"label": "neutral", "confidence": 0.77}


def test_disabled_local_sentiment_service_fails_closed():
    service = SentimentService(
        Settings(_env_file=None, DEBUG=False, HF_SENTIMENT_ENABLED=False)
    )
    with pytest.raises(SentimentServiceError) as raised:
        asyncio.run(service.analyze("A review"))
    assert raised.value.status_code == 503
    assert raised.value.code == "sentiment_disabled"


def test_sentiment_endpoint_response(sentiment_override):
    response = client.post(
        "/api/v1/sentiment/analyze", json={"review": "  This is a great purchase.  "}
    )
    assert response.status_code == 200
    assert response.json()["sentiment"] == {"label": "positive", "confidence": 0.91}
    assert response.json()["model"]["provider"] == "huggingface"
    assert response.json()["processing_time_ms"] >= 0


@pytest.mark.parametrize("review", ["", "   "])
def test_sentiment_endpoint_rejects_empty_input(review):
    response = client.post("/api/v1/sentiment/analyze", json={"review": review})
    assert response.status_code == 422


def test_sentiment_endpoint_enforces_configured_length(sentiment_override, monkeypatch):
    from app.routes import sentiment as sentiment_routes

    monkeypatch.setattr(sentiment_routes.settings, "HF_SENTIMENT_MAX_LENGTH", 5)
    response = client.post(
        "/api/v1/sentiment/analyze", json={"review": "x" * 6}
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "review_too_long"


def test_batch_sentiment_failure_is_isolated(sentiment_override):
    response = client.post(
        "/api/v1/sentiment/analyze/batch",
        json={"reviews": ["A good item", "A bad item"]},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["sentiment"]["label"] == "negative"
    assert results[1]["available"] is False
    assert results[1]["sentiment"] is None
    assert results[1]["error"] == "review_too_long"


def test_authenticity_contract_is_unchanged():
    class StubReviewService:
        pipeline = None

        def predict_review(self, review, rating, category, title="", db=None):
            return {
                "title": title or "Untitled Review",
                "prediction": 0,
                "review_type": "Human-Written Review",
                "confidence": "high",
                "category": category,
                "rating": rating,
                "clean_review": "cleaned",
                "reason": "existing behavior",
            }

    app.dependency_overrides[get_review_service] = StubReviewService
    app.dependency_overrides[get_db] = lambda: None
    try:
        response = client.post(
            "/api/v1/review/predict",
            json={
                "review": "A sufficiently long private review",
                "rating": 4,
                "category": "books",
            },
        )
    finally:
        app.dependency_overrides.pop(get_review_service, None)
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    assert set(response.json()) == {
        "title", "prediction", "review_type", "confidence", "category",
        "rating", "clean_review", "reason",
    }
