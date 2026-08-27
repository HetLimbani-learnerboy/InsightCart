"""
Project : InsightCart

File : metrics.py

Purpose :
Prometheus metrics for InsightCart.
"""

from prometheus_client import Counter, Histogram


# ============================================================
# HTTP REQUEST METRICS
# ============================================================

REQUEST_COUNT = Counter(
    "insightcart_http_requests_total",
    "Total number of HTTP requests",
    [
        "method",
        "endpoint",
        "status_code",
    ],
)


# ============================================================
# API LATENCY
# ============================================================

REQUEST_LATENCY = Histogram(
    "insightcart_http_request_duration_seconds",
    "HTTP request latency in seconds",
    [
        "method",
        "endpoint",
    ],
)


# ============================================================
# PREDICTION COUNT
# ============================================================

PREDICTION_COUNT = Counter(
    "insightcart_predictions_total",
    "Total number of review predictions",
    [
        "prediction_type",
    ],
)


# ============================================================
# ERROR COUNT
# ============================================================

ERROR_COUNT = Counter(
    "insightcart_http_errors_total",
    "Total number of HTTP errors",
    [
        "method",
        "endpoint",
        "status_code",
    ],
)