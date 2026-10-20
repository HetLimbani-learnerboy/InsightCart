import json
from pathlib import Path

from app.schemas.request import BatchPredictionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_extension_batch_request_matches_fastapi_contract():
    extension_dir = PROJECT_ROOT / "chrome_extension"
    manifest = json.loads((extension_dir / "manifest.json").read_text())
    background = (extension_dir / "background.js").read_text()

    assert "http://localhost:8000/*" in manifest["host_permissions"]
    assert 'const API_BASE_URL = "http://localhost:8000/api/v1";' in background
    assert "`${API_BASE_URL}/review/predict/batch`" in background
    assert "reviews: message.reviews" in background
    assert set(BatchPredictionRequest.model_fields) == {"reviews"}
    assert set(
        BatchPredictionRequest.model_fields["reviews"]
        .annotation.__args__[0]
        .model_fields
    ) >= {
        "review",
        "rating",
        "category",
    }


def test_compose_api_uses_docker_reachable_mlflow_uri():
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text()

    assert "MLFLOW_DOCKER_TRACKING_URI:-http://host.docker.internal:5000" in compose
    assert "MLFLOW_TRACING_ENABLED:-true" in compose
