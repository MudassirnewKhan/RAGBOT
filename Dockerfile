# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files and keeps stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install curl for container healthchecks and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user with a home directory
ARG UID=10001
RUN useradd -m -u "${UID}" appuser && \
    mkdir -p /app/.cache /app/faiss_store && \
    chown -R appuser:appuser /app

# Download dependencies (using our requirement.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Switch to the non-privileged user
USER appuser

# Copy application source code
COPY --chown=appuser:appuser . .

# Expose port 8000
EXPOSE 8000

# Docker healthcheck pinging our REST API /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["python", "run_api.py"]