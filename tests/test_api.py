from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.database.connection import get_db
from app.routes.review_detection import get_review_service

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200


def test_health():
    response = client.get("/health")

    assert response.status_code == 200


def test_metrics():
    response = client.get("/metrics")

    assert response.status_code == 200


def test_extension_origin_can_read_trace_id_header(
    monkeypatch, prediction_dependencies
):
    monkeypatch.setenv("MLFLOW_TRACING_ENABLED", "false")
    response = client.post(
        "/api/v1/review/predict/batch",
        json={
            "reviews": [
                {
                    "review": "Extension CORS test review",
                    "rating": 4,
                    "category": "books",
                }
            ]
        },
        headers={
            "Origin": "chrome-extension://insightcart-test",
        },
    )

    assert response.status_code == 200
    assert (
        "x-mlflow-trace-id"
        in response.headers.get("access-control-expose-headers", "").lower()
    )


class StubReviewService:
    def predict_review(self, review, rating, category, title="", db=None):
        return {
            "title": title or "Untitled Review",
            "prediction": 0,
            "review_type": "Human-Written Review",
            "confidence": "high",
            "category": category,
            "rating": rating,
            "clean_review": "cleaned response text",
            "reason": "safe response reason",
        }

    def predict_batch(self, reviews, db=None):
        results = [
            self.predict_review(
                item.review, item.rating, item.category, item.title or ""
            )
            for item in reviews
        ]
        return {"total_reviews": len(results), "results": results}


@pytest.fixture
def prediction_dependencies():
    app.dependency_overrides[get_review_service] = StubReviewService
    app.dependency_overrides[get_db] = lambda: None
    yield
    app.dependency_overrides.pop(get_review_service, None)
    app.dependency_overrides.pop(get_db, None)


@pytest.mark.parametrize("tracing_enabled", ["true", "false"])
def test_prediction_endpoints_preserve_responses_with_tracing_enabled_or_disabled(
    monkeypatch, prediction_dependencies, tracing_enabled
):
    monkeypatch.setenv("MLFLOW_TRACING_ENABLED", tracing_enabled)
    client = TestClient(app)

    single = client.post(
        "/api/v1/review/predict",
        json={
            "review": "A sufficiently long private review",
            "rating": 4,
            "category": "books",
        },
    )
    batch = client.post(
        "/api/v1/review/predict/batch",
        json={
            "reviews": [
                {"review": "Another private review", "rating": 5, "category": "books"}
            ]
        },
    )

    assert single.status_code == 200
    assert single.json()["prediction"] == 0
    assert single.json()["clean_review"] == "cleaned response text"
    assert batch.status_code == 200
    assert batch.json()["total_reviews"] == 1
    assert batch.json()["results"][0]["review_type"] == "Human-Written Review"


def test_prediction_succeeds_when_mlflow_span_creation_fails(
    monkeypatch, prediction_dependencies
):
    monkeypatch.setenv("MLFLOW_TRACING_ENABLED", "true")

    def fail_span(*args, **kwargs):
        raise RuntimeError("tracking server unavailable")

    monkeypatch.setattr("app.tracing.mlflow.start_span", fail_span)
    response = TestClient(app).post(
        "/api/v1/review/predict",
        json={
            "review": "A sufficiently long private review",
            "rating": 4,
            "category": "books",
        },
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 0
