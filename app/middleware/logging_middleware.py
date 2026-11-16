"""
Structured Request/Response Logging and Tracing Middleware.
Injects correlation IDs (X-Request-ID) and processing latency headers (X-Process-Time).
"""
import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# Configure structured application logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ragbot.api.access")


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """ASGI Middleware capturing request telemetry and response timings."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Extract or generate correlation ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        start_time = time.perf_counter()

        # Log incoming request
        client_ip = request.client.host if request.client else "unknown"
        logger.info(
            "START | req_id=%s | method=%s | path=%s | client=%s",
            request_id,
            request.method,
            request.url.path,
            client_ip,
        )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "FAIL  | req_id=%s | method=%s | path=%s | duration=%.2fms",
                request_id,
                request.method,
                request.url.path,
                duration_ms,
            )
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Inject tracing headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"

        # Log completion
        logger.info(
            "DONE  | req_id=%s | status=%d | duration=%.2fms | method=%s | path=%s",
            request_id,
            response.status_code,
            duration_ms,
            request.method,
            request.url.path,
        )

        return response
