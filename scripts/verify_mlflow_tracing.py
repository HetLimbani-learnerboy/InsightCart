"""Verify that a real API prediction trace is queryable from MLflow."""

import argparse
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mlflow

from app.tracing import EXPERIMENT_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-id",
        help="Match the X-MLflow-Trace-Id returned by a real prediction request",
    )
    parser.add_argument(
        "--route",
        choices=("single", "batch", "any"),
        default="any",
        help="Require the corresponding API route span (default: any)",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=15,
        help="Seconds to wait for the API's asynchronous trace export (default: 15)",
    )
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment was not found in {mlflow.get_tracking_uri()}: "
            f"{EXPERIMENT_NAME}"
        )

    expected_names = {
        "single": {"review.predict"},
        "batch": {"review.predict.batch"},
        "any": {"review.predict", "review.predict.batch"},
    }[args.route]

    deadline = time.monotonic() + max(args.wait_seconds, 0)
    matching_trace = None
    while True:
        traces = mlflow.search_traces(
            locations=[experiment.experiment_id],
            max_results=100,
            return_type="list",
            flush=True,
        )
        matching_trace = next(
            (
                trace
                for trace in traces
                if (args.trace_id is None or trace.info.request_id == args.trace_id)
                and expected_names.intersection(span.name for span in trace.data.spans)
            ),
            None,
        )
        if matching_trace is not None or time.monotonic() >= deadline:
            break
        time.sleep(min(0.5, max(deadline - time.monotonic(), 0)))
    if matching_trace is None:
        requested = f" trace_id={args.trace_id}" if args.trace_id else ""
        raise RuntimeError(
            "No queryable API prediction trace found in the configured MLflow "
            f"experiment={EXPERIMENT_NAME} route={args.route}{requested}. "
            "Send a real prediction request first; this verifier does not create "
            "synthetic traces."
        )

    span_names = [span.name for span in matching_trace.data.spans]
    print(f"tracking_uri={mlflow.get_tracking_uri()}")
    print(f"experiment_name={experiment.name}")
    print(f"experiment_id={experiment.experiment_id}")
    print(f"trace_id={matching_trace.info.request_id}")
    print(f"span_names={span_names}")
    print("api_trace_query_verified=true")


if __name__ == "__main__":
    main()
