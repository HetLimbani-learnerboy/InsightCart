"""Create and query a harmless MLflow trace for the configured tracking store."""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mlflow
from mlflow.entities import SpanType

from app.tracing import EXPERIMENT_NAME, configure_tracing


def main() -> None:
    configure_tracing()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment was not created: {EXPERIMENT_NAME}")

    @mlflow.trace(name="diagnostic.synthetic", span_type=SpanType.CHAIN)
    def synthetic_operation(left: int, right: int) -> dict:
        active_span = mlflow.get_current_active_span()
        return {"result": left + right, "trace_id": active_span.trace_id}

    result = synthetic_operation(19, 23)
    traces = mlflow.search_traces(
        locations=[experiment.experiment_id],
        max_results=100,
        return_type="list",
        flush=True,
    )
    matching_trace = next(
        (trace for trace in traces if trace.info.request_id == result["trace_id"]),
        None,
    )
    if matching_trace is None:
        raise RuntimeError(
            f"Trace {result['trace_id']} was not queryable from the configured store"
        )

    span_names = [span.name for span in matching_trace.data.spans]
    if "diagnostic.synthetic" not in span_names:
        raise RuntimeError(f"Expected synthetic span missing: {span_names}")

    print(f"tracking_uri={mlflow.get_tracking_uri()}")
    print(f"experiment_name={experiment.name}")
    print(f"experiment_id={experiment.experiment_id}")
    print(f"trace_id={result['trace_id']}")
    print(f"span_names={span_names}")
    print("trace_query_verified=true")


if __name__ == "__main__":
    main()
