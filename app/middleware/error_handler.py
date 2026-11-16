"""
Global Exception Handlers and RFC 7807 Error Response formatting.
"""
from datetime import datetime
import logging
from fastapi import Request, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import groq

logger = logging.getLogger("ragbot.error_handler")


def create_error_response(status_code: int, error_type: str, message: str, path: str, details=None) -> JSONResponse:
    """Construct a consistent, RFC 7807-compliant JSON error envelope."""
    payload = {
        "status_code": status_code,
        "error_type": error_type,
        "message": message,
        "path": path,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if details:
        payload["details"] = details
    return JSONResponse(status_code=status_code, content=payload)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handler for Pydantic input schema validation failures (HTTP 422)."""
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    formatted_errors = []
    for err in exc.errors():
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        formatted_errors.append({
            "field": loc,
            "message": err.get("msg", "Invalid input"),
            "type": err.get("type", "validation_error"),
        })

    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_type="VALIDATION_ERROR",
        message="Request payload failed schema validation.",
        path=request.url.path,
        details=formatted_errors,
    )


async def groq_ratelimit_exception_handler(request: Request, exc: groq.RateLimitError) -> JSONResponse:
    """Handler for upstream Groq 429 Rate Limit exhaustion."""
    logger.error("Groq rate limit exceeded on %s: %s", request.url.path, exc)
    return create_error_response(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        error_type="UPSTREAM_RATE_LIMIT",
        message="The LLM inference endpoint is temporarily rate-limited. Please retry shortly.",
        path=request.url.path,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handler for standard FastAPI HTTPExceptions."""
    return create_error_response(
        status_code=exc.status_code,
        error_type="HTTP_ERROR",
        message=exc.detail,
        path=request.url.path,
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled internal server errors (HTTP 500)."""
    logger.exception("Unhandled server exception on %s: %s", request.url.path, exc)
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_type="INTERNAL_SERVER_ERROR",
        message="An unexpected server error occurred while processing the request.",
        path=request.url.path,
    )
