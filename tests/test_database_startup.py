from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from types import SimpleNamespace

import app.database.init_db as database_init
import app.main as main_module


def test_init_db_creates_prediction_table_in_empty_database(tmp_path, monkeypatch):
    database_path = tmp_path / "fresh.db"
    engine = create_engine(f"sqlite:///{database_path}")
    monkeypatch.setattr(database_init, "engine", engine)

    database_init.init_db()

    assert "review_predictions" in inspect(engine).get_table_names()
    engine.dispose()


def test_application_lifespan_initializes_database(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'startup.db'}")
    monkeypatch.setattr(database_init, "engine", engine)

    with TestClient(main_module.app):
        pass

    assert "review_predictions" in inspect(engine).get_table_names()
    engine.dispose()


def test_batch_prediction_persists_after_initializing_fresh_database(
    tmp_path, monkeypatch
):
    from app.database.models import ReviewPrediction
    from app.services.review_service import ReviewDetectionService

    engine = create_engine(f"sqlite:///{tmp_path / 'batch.db'}")
    monkeypatch.setattr(database_init, "engine", engine)
    database_init.init_db()
    session = sessionmaker(bind=engine)()

    class StubPipeline:
        def predict(self, **kwargs):
            return {
                "title": "Test review",
                "prediction": 0,
                "review_type": "Human-Written Review",
                "confidence": "high",
                "category": "books",
                "rating": 5,
                "clean_review": "clean review",
                "reason": "fixture",
            }

    service = object.__new__(ReviewDetectionService)
    service.pipeline = StubPipeline()
    request = SimpleNamespace(
        review="This is a test product review",
        title="Test review",
        rating=5,
        category="books",
    )

    result = service.predict_batch([request], db=session)

    assert result["total_reviews"] == 1
    assert session.query(ReviewPrediction).count() == 1
    session.close()
    engine.dispose()
