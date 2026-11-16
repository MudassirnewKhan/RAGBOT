"""
RAGBOT Production FastAPI Application Entrypoint.
Configures Lifespan, Tracing & Logging Middleware, CORS, Error Handlers, and API Routers.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
import groq

from app.config import settings
from app.middleware.logging_middleware import StructuredLoggingMiddleware
from app.middleware.error_handler import (
    validation_exception_handler,
    groq_ratelimit_exception_handler,
    http_exception_handler,
    generic_exception_handler,
)
from app.routers import health, query, documents
from app.dependencies import set_rag_engine
from src.rag_engine import RAGEngine

logger = logging.getLogger("ragbot.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Context Manager.
    Pre-warms the singleton RAGEngine during server startup so models load once.
    """
    logger.info("Initializing RAG Engine singleton during startup...")
    try:
        engine = RAGEngine()
        set_rag_engine(engine)
        app.state.rag_engine = engine
        logger.info("RAG Engine successfully initialized and ready for inference.")
    except Exception as exc:
        logger.error("Failed to initialize RAG Engine during startup: %s", exc)
        app.state.rag_engine = None

    yield

    logger.info("Gracefully shutting down RAGBOT application.")
    app.state.rag_engine = None


def create_application() -> FastAPI:
    """FastAPI application factory."""
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # 1. Attach Structured Logging Middleware (attaches X-Request-ID and X-Process-Time)
    application.add_middleware(StructuredLoggingMiddleware)

    # 2. Configure Cross-Origin Resource Sharing (CORS) with exposed headers
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # 3. Register Global RFC 7807 Exception Handlers
    application.add_exception_handler(RequestValidationError, validation_exception_handler)
    application.add_exception_handler(groq.RateLimitError, groq_ratelimit_exception_handler)
    application.add_exception_handler(HTTPException, http_exception_handler)
    application.add_exception_handler(Exception, generic_exception_handler)

    # 4. Mount Endpoint Routers
    application.include_router(health.router)
    application.include_router(query.router)
    application.include_router(documents.router)

    @application.get("/", tags=["Root"])
    async def root():
        return {
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "status": "active",
            "docs_url": "/docs",
        }

    return application


app = create_application()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
