"""Privacy-conscious, best-effort MLflow tracing for API predictions."""

from contextlib import contextmanager
import os
import time
from typing import Any, Dict, Iterator, Optional

import mlflow
from mlflow.entities import SpanStatus, SpanStatusCode, SpanType

from src.logger import logger

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "InsightCart-API-Tracing"


def tracing_enabled() -> bool:
    return os.getenv("MLFLOW_TRACING_ENABLED", "true").lower() not in {
        "0",
        "false",
        "no",
    }


def span_trace_id(span: Any) -> Optional[str]:
    return getattr(span, "trace_id", None) if span is not None else None


def configure_tracing() -> None:
    """Configure the API's tracking destination without blocking predictions."""
    if not tracing_enabled():
        logger.info("MLflow tracing disabled")
        return

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info("MLflow tracing configured experiment=%s", EXPERIMENT_NAME)
    except Exception as exc:  # Tracing setup must never prevent API startup.
        logger.warning(
            "MLflow tracing setup unavailable error_type=%s", type(exc).__name__
        )


@contextmanager
def traced_span(
    name: str,
    inputs: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """Create a span while ensuring MLflow errors cannot affect application flow."""
    if not tracing_enabled():
        yield None
        return

    manager = None
    span = None
    started = time.perf_counter()
    try:
        manager = mlflow.start_span(name=name, span_type=SpanType.CHAIN)
        span = manager.__enter__()
        logger.info(
            "MLflow span started name=%s trace_id=%s",
            name,
            span_trace_id(span) or "unavailable",
        )
        if inputs:
            span.set_inputs(inputs)
    except Exception as exc:
        if manager is not None:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                pass
        logger.warning(
            "MLflow span start failed name=%s error_type=%s",
            name,
            type(exc).__name__,
        )
        span = None
        manager = None

    try:
        yield span
    except BaseException as exc:
        if span is not None:
            try:
                span.set_status(
                    SpanStatus(
                        status_code=SpanStatusCode.ERROR,
                        description=type(exc).__name__,
                    )
                )
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
            logger.info(
                "MLflow span closed name=%s trace_id=%s status=error",
                name,
                span_trace_id(span) or "unavailable",
            )
        raise
    else:
        if span is not None:
            try:
                span.set_attribute("status", "success")
                span.set_attribute(
                    "execution_duration_ms", (time.perf_counter() - started) * 1000
                )
                manager.__exit__(None, None, None)
                logger.info(
                    "MLflow span closed name=%s trace_id=%s status=success",
                    name,
                    span_trace_id(span) or "unavailable",
                )
            except Exception as exc:
                logger.warning(
                    "MLflow span finish failed name=%s error_type=%s",
                    name,
                    type(exc).__name__,
                )


def set_span_outputs(span: Any, outputs: Dict[str, Any]) -> None:
    """Attach non-sensitive output metadata without affecting the caller."""
    if span is None:
        return
    try:
        span.set_outputs(outputs)
    except Exception as exc:
        logger.debug("MLflow span output failed: %s", type(exc).__name__)
