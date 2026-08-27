"""
Project : InsightCart

File : prometheus.py

Purpose :
Collect HTTP request metrics for Prometheus.
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ERROR_COUNT,
)


class PrometheusMiddleware(BaseHTTPMiddleware):

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):

        start_time = time.perf_counter()

        response = None

        try:

            response = await call_next(request)

            return response

        finally:

            duration = time.perf_counter() - start_time

            method = request.method

            endpoint = request.url.path

            status_code = (
                response.status_code
                if response is not None
                else 500
            )

            # Request count
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code),
            ).inc()

            # Request latency
            REQUEST_LATENCY.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)

            # Errors
            if status_code >= 400:

                ERROR_COUNT.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=str(status_code),
                ).inc()