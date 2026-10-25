import joblib
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer
from src.components.model_pusher import ModelPusher, ModelPusherConfig
from src.evaluation.model_gate import model_quality_gate


class FixedClassifier:
    def __init__(self, predictions=None, fail=False):
        self.predictions = predictions
        self.fail = fail

    def fit(self, features, labels):
        if self.fail:
            raise ValueError("intentional test failure")
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, features):
        return np.resize(self.predictions, features.shape[0])

    def predict_proba(self, features):
        positive = np.resize(self.predictions, features.shape[0]).astype(float)
        return np.column_stack((1 - positive, positive))


def test_numeric_binary_targets_validate():
    dataframe = pd.DataFrame(
        {
            "category": ["books", "games"],
            "rating": [4, 5],
            "label": [0, 1],
            "text_": ["good book", "fun game"],
        }
    )

    validated = DataValidation().validate_dataset(dataframe)

    assert validated["label"].tolist() == [0, 1]


def test_invalid_targets_are_rejected():
    dataframe = pd.DataFrame(
        {
            "category": ["books"],
            "rating": [4],
            "label": ["unknown"],
            "text_": ["review"],
        }
    )

    with pytest.raises(Exception, match="Expected both target classes"):
        DataValidation().validate_dataset(dataframe)


def test_single_class_dataset_is_rejected():
    dataframe = pd.DataFrame(
        {
            "category": ["books", "games"],
            "rating": [4, 5],
            "label": [0, 0],
            "text_": ["book review", "game review"],
        }
    )

    with pytest.raises(Exception, match="Expected both target classes"):
        DataValidation().validate_dataset(dataframe)


def test_validation_drops_incomplete_and_duplicate_rows_before_splitting():
    dataframe = pd.DataFrame(
        {
            "category": ["books", "books", "games", "games"],
            "rating": [4, 4, 5, 5],
            "label": [0, 0, 1, None],
            "text_": ["same review", "same review", "game review", "incomplete"],
        }
    )

    validated = DataValidation().validate_dataset(dataframe)

    assert len(validated) == 2
    assert validated["text_"].tolist() == ["same review", "game review"]


def test_best_model_is_selected_by_validation_and_keeps_its_estimator():
    trainer = ModelTrainer()
    winner = FixedClassifier([0, 1, 1, 0])
    trainer.models = {
        "Logistic Regression": FixedClassifier([0, 0, 0, 0]),
        "Random Forest": winner,
    }
    features = np.ones((4, 2))
    labels = np.array([0, 1, 1, 0])

    artifacts = trainer.initiate_model_training(features, features, labels, labels)

    assert artifacts.best_model_name == "Random Forest"
    assert artifacts.trained_model is artifacts.trained_models["Random Forest"]
    assert artifacts.validation_metrics["F1 Score"] == 1.0
    assert artifacts.comparison_dataframe["Selected"].sum() == 1


def test_failed_model_is_reported_without_hiding_successful_candidates():
    trainer = ModelTrainer()
    trainer.models = {
        "Broken Candidate": FixedClassifier(fail=True),
        "Working Candidate": FixedClassifier([0, 1]),
    }
    features = np.ones((2, 1))
    labels = np.array([0, 1])

    artifacts = trainer.initiate_model_training(features, features, labels, labels)

    failed = artifacts.comparison_dataframe.set_index("Model").loc["Broken Candidate"]
    assert failed["Status"] == "failed"
    assert "intentional test failure" in failed["Error"]
    assert artifacts.best_model_name == "Working Candidate"


def test_missing_model_dependency_is_reported_as_failed_candidate():
    trainer = ModelTrainer()
    trainer.models = {
        "Unavailable Candidate": None,
        "Working Candidate": FixedClassifier([0, 1]),
    }
    trainer.dependency_errors = {
        "Unavailable Candidate": "optional package is not installed"
    }
    features = np.ones((2, 1))
    labels = np.array([0, 1])

    artifacts = trainer.initiate_model_training(features, features, labels, labels)

    failure = artifacts.comparison_dataframe.set_index("Model").loc[
        "Unavailable Candidate"
    ]
    assert failure["Status"] == "failed"
    assert "optional package is not installed" in failure["Error"]


def test_quality_gate_requires_matching_evaluation_and_candidate_improvement(
    monkeypatch,
):
    monkeypatch.setenv("MODEL_MIN_PRECISION", "0.90")
    production = {
        "accuracy": 0.90,
        "precision": 0.90,
        "recall": 0.90,
        "f1": 0.90,
        "evaluation_dataset_id": "heldout-1",
    }
    candidate = {**production, "f1": 0.91}

    assert model_quality_gate(candidate, production)
    assert not model_quality_gate(production, production)
    assert not model_quality_gate(candidate, None)
    assert not model_quality_gate(
        {**candidate, "evaluation_dataset_id": "different-heldout"}, production
    )
    assert not model_quality_gate({**candidate, "precision": 0.89}, production)
    assert not model_quality_gate({**candidate, "f1": float("nan")}, production)


def test_preprocessors_and_splits_can_be_written_only_to_experiment_directory(
    tmp_path,
):
    rows = []
    for index in range(40):
        rows.append(
            {
                "category": "books" if index % 2 else "games",
                "rating": index % 5 + 1,
                "label": index % 2,
                "text_": f"review token{index % 3} quality",
                "review_length": 100000 + index,
            }
        )
    dataframe = DataValidation().validate_dataset(pd.DataFrame(rows))
    paths = {
        name: str(tmp_path / name)
        for name in ("vectorizer.pkl", "encoder.pkl", "scaler.pkl")
    }
    config = DataTransformationConfig(
        max_features=100,
        min_df=1,
        max_df=1.0,
        tfidf_path=paths["vectorizer.pkl"],
        encoder_path=paths["encoder.pkl"],
        scaler_path=paths["scaler.pkl"],
    )

    result = DataTransformation(config).initiate_data_transformation(dataframe)

    assert result.X_train.shape[0] == 24
    assert result.X_validation.shape[0] == 4
    assert result.X_promotion.shape[0] == 4
    assert result.X_test.shape[0] == 8
    assert len(result.promotion_dataset_id) == 64
    assert len(result.test_dataset_id) == 64
    assert all((tmp_path / name).exists() for name in paths)
    assert result.X_train.shape[1] == len(result.tfidf.vocabulary_) + 2 + 1


def test_selected_model_bundle_round_trips(tmp_path):
    source_bundle = {
        "model": FixedClassifier([1]),
        "model_name": "Dynamic Candidate",
        "tfidf": "vectorizer",
        "category_encoder": "encoder",
        "rating_scaler": "scaler",
    }
    model_path = tmp_path / "selected_model.pkl"
    joblib.dump(source_bundle, model_path)

    loaded_bundle = joblib.load(model_path)

    assert loaded_bundle["model_name"] == "Dynamic Candidate"
    assert loaded_bundle["model"].predict(np.ones((1, 1))).tolist() == [1]


def test_promoted_bundle_is_the_bundle_loaded_for_inference(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = FixedClassifier([1])
    trainer_artifacts = SimpleNamespace(
        trained_model=model, best_model_name="Chosen Candidate"
    )
    transform_artifacts = SimpleNamespace(
        tfidf="vectorizer", category_encoder="encoder", rating_scaler="scaler"
    )
    pusher = ModelPusher()
    pusher.config = ModelPusherConfig(
        model_path=str(model_dir / "legacy.pkl"),
        tfidf_path=str(model_dir / "tfidf.pkl"),
        encoder_path=str(model_dir / "encoder.pkl"),
        scaler_path=str(model_dir / "scaler.pkl"),
    )

    pushed_path = pusher.initiate_model_pusher(
        trainer_artifacts, transformation_artifacts=transform_artifacts
    )
    monkeypatch.setattr(
        "src.pipeline.review_detection_inference_pipeline.MODEL_DIR", str(model_dir)
    )

    from src.pipeline.review_detection_inference_pipeline import (
        ReviewDetectionInferencePipeline,
    )

    loaded_pipeline = ReviewDetectionInferencePipeline()

    assert pushed_path == str(model_dir / "production_model_bundle.pkl")
    assert loaded_pipeline.model.predict(np.ones((1, 1))).tolist() == [1]
    assert loaded_pipeline.vectorizer == "vectorizer"
    assert loaded_pipeline.category_encoder == "encoder"
    assert loaded_pipeline.rating_scaler == "scaler"
    assert loaded_pipeline.model_version == "Chosen Candidate"


@pytest.mark.parametrize(
    ("promote", "gate_passed", "failure_stage"),
    [
        (False, False, None),
        (True, False, None),
        (True, True, None),
        (True, True, "production_evaluation"),
        (True, True, "training"),
        (True, True, "evaluation"),
    ],
)
def test_v2_experiment_promotes_only_when_explicit_and_approved(
    tmp_path, monkeypatch, promote, gate_passed, failure_stage
):
    from src.pipeline import run_v2_experiment as experiment

    dataset = pd.DataFrame(
        {
            "category": ["books", "games"],
            "rating": [4, 5],
            "label": [0, 1],
            "text_": ["book review", "game review"],
            "review_length": [12, 12],
        }
    )
    project_root = tmp_path / "project"
    production_dir = project_root / "artifacts"
    production_dir.mkdir(parents=True)
    production_model_dir = production_dir / "models"
    production_model_dir.mkdir()
    production_bundle = production_model_dir / "production_model_bundle.pkl"
    production_bundle.write_bytes(b"existing-production-bundle")
    original_bundle = production_bundle.read_bytes()
    production_metrics = production_dir / "production_metrics.json"
    production_metrics.write_text('{"f1": 0.9}', encoding="utf-8")
    original_metrics = production_metrics.read_bytes()
    output_dir = tmp_path / "artifacts" / "experiments" / "v2"

    class FakeIngestion:
        def __init__(self, config):
            self.config = config

        def initiate_data_ingestion(self):
            return dataset

    class FakeValidation:
        def validate_dataset(self, dataframe):
            return dataframe

    class FakeTransformation:
        def __init__(self, config):
            self.config = config

        def initiate_data_transformation(self, dataframe):
            assert str(output_dir) in self.config.tfidf_path
            assert str(output_dir) in self.config.encoder_path
            assert str(output_dir) in self.config.scaler_path
            return SimpleNamespace(
                X_train=np.ones((2, 1)),
                X_validation=np.ones((2, 1)),
                X_promotion=np.ones((2, 1)),
                X_test=np.ones((2, 1)),
                y_train=np.array([0, 1]),
                y_validation=np.array([0, 1]),
                y_promotion=np.array([0, 1]),
                y_test=np.array([0, 1]),
                tfidf="vectorizer",
                category_encoder="encoder",
                rating_scaler="scaler",
                promotion_dataset_id="heldout-fixture",
                test_dataset_id="test-fixture",
                promotion_dataframe=dataset,
                test_dataframe=dataset,
            )

    class FakeTrainer:
        def __init__(self, config):
            self.config = config
            self.models = {"Candidate": object()}

        def initiate_model_training(self, *args):
            if failure_stage == "training":
                raise RuntimeError("training fixture failure")
            pd.DataFrame([{"Model": "Candidate", "Status": "success"}]).to_csv(
                self.config.comparison_path, index=False
            )
            return SimpleNamespace(
                trained_model=FixedClassifier([1]),
                best_model_name="Candidate",
                validation_metrics={
                    "Accuracy": 1.0,
                    "Precision": 1.0,
                    "Recall": 1.0,
                    "F1 Score": 1.0,
                    "ROC AUC": 1.0,
                },
            )

    class FakeEvaluation:
        def initiate_model_evaluation(self, *args, metrics_path, **kwargs):
            if failure_stage == "evaluation":
                raise RuntimeError("evaluation fixture failure")
            metrics_path_obj = tmp_path / metrics_path
            metrics_path_obj.parent.mkdir(parents=True, exist_ok=True)
            metrics_path_obj.write_text("{}", encoding="utf-8")
            return SimpleNamespace(
                metrics={
                    "Accuracy": 1.0,
                    "Precision": 1.0,
                    "Recall": 1.0,
                    "F1 Score": 1.0,
                    "ROC AUC": 1.0,
                }
            )

    monkeypatch.setattr(experiment, "project_root", str(project_root))
    monkeypatch.setattr(experiment, "DataIngestion", FakeIngestion)
    monkeypatch.setattr(experiment, "DataValidation", FakeValidation)
    monkeypatch.setattr(experiment, "DataTransformation", FakeTransformation)
    monkeypatch.setattr(experiment, "ModelTrainer", FakeTrainer)
    monkeypatch.setattr(experiment, "ModelEvaluation", FakeEvaluation)
    production_metrics_fixture = {
        "accuracy": 0.90,
        "precision": 0.90,
        "recall": 0.90,
        "f1": 0.90,
        "evaluation_dataset_id": "heldout-fixture",
        "model_version": "production-fixture-v1",
    }

    def evaluate_production_fixture(dataframe):
        if failure_stage == "production_evaluation":
            raise RuntimeError("production_evaluation fixture failure")
        return dict(production_metrics_fixture)

    monkeypatch.setattr(
        experiment,
        "evaluate_production_model",
        evaluate_production_fixture,
    )
    promotion_calls = []

    class FakePusher:
        def initiate_model_pusher(self, *args, **kwargs):
            promotion_calls.append((args, kwargs))
            return "fixture-production-bundle"

    monkeypatch.setattr(experiment, "ModelPusher", FakePusher)
    monkeypatch.setattr(
        experiment,
        "evaluate_model_quality",
        lambda *args, **kwargs: (
            gate_passed,
            "fixture approval" if gate_passed else "fixture rejection",
        ),
    )
    monkeypatch.setattr(experiment, "package_versions", lambda: {})

    dataset_path = tmp_path / "synthetic.csv"
    dataset_path.write_text("fixture", encoding="utf-8")
    if failure_stage:
        with pytest.raises(RuntimeError, match=f"{failure_stage} fixture failure"):
            experiment.run_experiment(
                str(dataset_path), str(output_dir), random_seed=7, promote=promote
            )
        assert not promotion_calls
        assert production_bundle.read_bytes() == original_bundle
        assert production_metrics.read_bytes() == original_metrics
        return

    _, _, metadata = experiment.run_experiment(
        str(dataset_path), str(output_dir), random_seed=7, promote=promote
    )

    expected_promotion = promote and gate_passed
    assert metadata["production_promotion_performed"] is expected_promotion
    assert len(promotion_calls) == int(expected_promotion)
    assert metadata["training_configuration"]["random_seed"] == 7
    assert production_bundle.read_bytes() == original_bundle
    assert production_metrics.read_bytes() == original_metrics
    assert (output_dir / "selected_model.pkl").exists()
    assert (output_dir / "model_metadata.json").exists()


def test_quality_gate_passes_only_with_configured_metric_floors_and_improvement():
    from src.evaluation.model_gate import evaluate_model_quality

    baseline = {
        "accuracy": 0.90,
        "precision": 0.90,
        "recall": 0.90,
        "f1": 0.90,
        "evaluation_dataset_id": "same-heldout-set",
    }
    candidate = {**baseline, "accuracy": 0.92, "f1": 0.92}
    thresholds = {metric: 0.85 for metric in ("accuracy", "precision", "recall", "f1")}

    passed, reason = evaluate_model_quality(
        candidate,
        baseline,
        thresholds=thresholds,
        minimum_improvement=0.01,
    )

    assert passed
    assert "passed" in reason
