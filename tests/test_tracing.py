import json

import mlflow
import pytest
from fastapi.testclient import TestClient

from app.database.connection import get_db
from app.main import app
from app.routes.review_detection import get_review_service
from app.tracing import EXPERIMENT_NAME, configure_tracing
from mlflow.entities import SpanStatusCode


class StubReviewService:
    def predict_review(self, review, rating, category, title="", db=None):
        return {
            "title": title or "Untitled Review",
            "prediction": 0,
            "review_type": "Human-Written Review",
            "confidence": "high",
            "category": category,
            "rating": rating,
            "clean_review": "safe cleaned result",
            "reason": "safe prediction reason",
        }

    def predict_batch(self, reviews, db=None):
        results = [
            self.predict_review(
                item.review, item.rating, item.category, item.title or ""
            )
            for item in reviews
        ]
        return {"total_reviews": len(results), "results": results}


class FailingReviewService:
    def predict_review(self, **kwargs):
        raise RuntimeError("TRACE_SECRET_SENTINEL")


@pytest.fixture
def local_tracing_store(tmp_path, monkeypatch):
    tracking_uri = f"sqlite:///{tmp_path / 'tracing.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_TRACING_ENABLED", "true")
    configure_tracing()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None
    yield experiment.experiment_id
    mlflow.set_tracking_uri("file:./mlruns")


@pytest.fixture
def api_client():
    app.dependency_overrides[get_review_service] = StubReviewService
    app.dependency_overrides[get_db] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.pop(get_review_service, None)
    app.dependency_overrides.pop(get_db, None)


def read_traces(experiment_id):
    return mlflow.search_traces(
        locations=[experiment_id],
        max_results=100,
        return_type="list",
        flush=True,
    )


def test_tracking_uri_uses_explicit_environment(monkeypatch, tmp_path):
    uri = f"sqlite:///{tmp_path / 'tracking-uri.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_TRACING_ENABLED", "true")

    configure_tracing()

    assert mlflow.get_tracking_uri() == uri


def test_single_prediction_creates_private_trace(local_tracing_store, api_client):
    private_review = "PRIVATE_REVIEW_SENTINEL do not store this sentence"
    response = api_client.post(
        "/api/v1/review/predict",
        json={"review": private_review, "rating": 4, "category": "books"},
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 0
    assert response.headers.get("X-MLflow-Trace-Id")
    traces = read_traces(local_tracing_store)
    trace = next(
        item
        for item in traces
        if any(span.name == "review.predict" for span in item.data.spans)
    )
    serialized = json.dumps(
        [
            {
                "name": span.name,
                "inputs": span.inputs,
                "outputs": span.outputs,
                "attributes": span.attributes,
            }
            for span in trace.data.spans
        ],
        default=str,
    )
    assert private_review not in serialized
    assert "review.response_generation" in serialized


def test_batch_prediction_creates_trace_with_count_only(
    local_tracing_store, api_client
):
    private_review = "BATCH_PRIVATE_SENTINEL private content"
    response = api_client.post(
        "/api/v1/review/predict/batch",
        json={
            "reviews": [{"review": private_review, "rating": 5, "category": "books"}]
        },
    )

    assert response.status_code == 200
    assert response.json()["total_reviews"] == 1
    assert response.headers.get("X-MLflow-Trace-Id")
    traces = read_traces(local_tracing_store)
    trace = next(
        item
        for item in traces
        if any(span.name == "review.predict.batch" for span in item.data.spans)
    )
    serialized = json.dumps([span.inputs for span in trace.data.spans], default=str)
    assert private_review not in serialized
    assert "review_count" in serialized


def test_extension_batch_payload_matches_backend_and_has_batch_trace(
    local_tracing_store, api_client
):
    extension_review = {
        "id": "amazon-review-1",
        "title": "Product title",
        "review": "An extension-compatible review body.",
        "rating": 4,
        "category": "books",
        "reviewer": "Reviewer name",
        "date": "January 1, 2026",
        "verified_purchase": True,
        "helpful_votes": 2,
    }
    response = api_client.post(
        "/api/v1/review/predict/batch",
        json={"reviews": [extension_review]},
    )

    assert response.status_code == 200
    assert response.json()["total_reviews"] == 1
    assert response.headers.get("X-MLflow-Trace-Id")
    trace = next(
        item
        for item in read_traces(local_tracing_store)
        if item.info.request_id == response.headers["X-MLflow-Trace-Id"]
    )
    assert any(span.name == "review.predict.batch" for span in trace.data.spans)


def test_invalid_request_and_tracing_failure_preserve_api_behavior(
    monkeypatch, local_tracing_store, api_client
):
    invalid_response = api_client.post(
        "/api/v1/review/predict",
        json={"review": "x", "rating": 9, "category": "books"},
    )
    assert invalid_response.status_code == 422

    def fail_start_span(*args, **kwargs):
        raise RuntimeError("tracking store unavailable")

    monkeypatch.setattr("app.tracing.mlflow.start_span", fail_start_span)
    valid_response = api_client.post(
        "/api/v1/review/predict",
        json={"review": "A harmless valid review", "rating": 4, "category": "books"},
    )
    assert valid_response.status_code == 200
    assert valid_response.json()["prediction"] == 0


def test_model_error_trace_does_not_capture_exception_message(
    local_tracing_store, api_client
):
    app.dependency_overrides[get_review_service] = FailingReviewService
    response = api_client.post(
        "/api/v1/review/predict",
        json={
            "review": "A valid request whose model will fail",
            "rating": 4,
            "category": "books",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed."
    assert response.headers.get("X-MLflow-Trace-Id")
    traces = read_traces(local_tracing_store)
    trace = next(
        item
        for item in traces
        if any(span.name == "review.predict" for span in item.data.spans)
    )
    serialized = json.dumps(
        [
            {
                "inputs": span.inputs,
                "outputs": span.outputs,
                "attributes": span.attributes,
            }
            for span in trace.data.spans
        ],
        default=str,
    )
    assert "TRACE_SECRET_SENTINEL" not in serialized
    assert "RuntimeError" in serialized
    assert trace.data.spans[0].status.status_code == SpanStatusCode.ERROR


def test_disabled_tracing_preserves_batch_api_without_trace_id(monkeypatch, api_client):
    monkeypatch.setenv("MLFLOW_TRACING_ENABLED", "false")
    response = api_client.post(
        "/api/v1/review/predict/batch",
        json={
            "reviews": [
                {
                    "review": "A disabled tracing review.",
                    "rating": 3,
                    "category": "books",
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["total_reviews"] == 1
    assert "X-MLflow-Trace-Id" not in response.headers
