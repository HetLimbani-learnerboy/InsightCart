"""Privacy-conscious, best-effort MLflow tracing for API predictions."""

from contextlib import contextmanager
import os
import time
from typing import Any, Dict, Iterator, Optional

import mlflow
from mlflow.entities import SpanType

from src.logger import logger

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "InsightCart-API-Tracing"


def configure_tracing() -> None:
    """Configure the API's tracking destination without blocking predictions."""
    if os.getenv("MLFLOW_TRACING_ENABLED", "true").lower() in {"0", "false", "no"}:
        return

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as exc:  # Tracing setup must never prevent API startup.
        logger.warning("MLflow tracing setup unavailable: %s", type(exc).__name__)


@contextmanager
def traced_span(
    name: str,
    inputs: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """Create a span while ensuring MLflow errors cannot affect application flow."""
    if os.getenv("MLFLOW_TRACING_ENABLED", "true").lower() in {"0", "false", "no"}:
        yield None
        return

    manager = None
    span = None
    started = time.perf_counter()
    try:
        manager = mlflow.start_span(name=name, span_type=SpanType.CHAIN)
        span = manager.__enter__()
        if inputs:
            span.set_inputs(inputs)
    except Exception as exc:
        if manager is not None:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                pass
        logger.debug("MLflow span start failed: %s", type(exc).__name__)
        span = None
        manager = None

    try:
        yield span
    except BaseException as exc:
        if span is not None:
            try:
                span.set_attribute("status", "error")
                span.set_attribute("error_type", type(exc).__name__)
                span.set_attribute(
                    "execution_duration_ms", (time.perf_counter() - started) * 1000
                )
            except Exception:
                pass
            try:
                manager.__exit__(None, None, None)
            except Exception:
                pass
        raise
    else:
        if span is not None:
            try:
                span.set_attribute("status", "success")
                span.set_attribute(
                    "execution_duration_ms", (time.perf_counter() - started) * 1000
                )
                manager.__exit__(None, None, None)
            except Exception as exc:
                logger.debug("MLflow span finish failed: %s", type(exc).__name__)


def set_span_outputs(span: Any, outputs: Dict[str, Any]) -> None:
    """Attach non-sensitive output metadata without affecting the caller."""
    if span is None:
        return
    try:
        span.set_outputs(outputs)
    except Exception as exc:
        logger.debug("MLflow span output failed: %s", type(exc).__name__)
