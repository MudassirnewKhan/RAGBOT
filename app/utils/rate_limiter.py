"""
Rate-limit and network resilience utility using Tenacity.
Provides exponential backoff with randomized jitter for Groq LLM API calls.
"""
import logging
from typing import Callable, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception_type,
    before_sleep_log,
)
import groq

logger = logging.getLogger("ragbot.rate_limiter")


def is_retryable_groq_error(exception: BaseException) -> bool:
    """Determine if a Groq API exception is transient and safe to retry."""
    return isinstance(
        exception,
        (
            groq.RateLimitError,          # HTTP 429 Too Many Requests
            groq.APIConnectionError,      # Transient network failure / timeout
            groq.InternalServerError,     # HTTP 500+ temporary server error
        )
    )


# Retry decorator: 4 attempts, exponential backoff (1s -> 2s -> 4s -> max 10s) + jitter
groq_retry = retry(
    retry=retry_if_exception_type((groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError)),
    wait=wait_exponential(multiplier=1, min=1, max=10) + wait_random(0, 1),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def execute_with_retry(func: Callable[..., Any], *args, **kwargs) -> Any:
    """Wrapper function executing a target callable with exponential backoff retry."""
    @groq_retry
    def _invoker():
        return func(*args, **kwargs)
    return _invoker()
