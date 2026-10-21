# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# Prevents .pyc files and enables unbuffered stdout for container logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build from inside kdd_app/ so this directory is the build context:
#   cd kdd_app; docker build -t kdd-ids-app .

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/outputs && chown -R appuser:appuser /app/outputs
USER appuser

ENV MODEL_DIR=/app/models/kdd

EXPOSE 5000

# Recommended resource limits (enforce at run/orchestration time, e.g.):
#   docker run --cpus="1.0" --memory="512m" --memory-swap="512m" \
#     -e CPU_LIMIT=1.0 -e MEM_LIMIT_MB=512 \
#     -v ${PWD}/outputs:/app/outputs \
#     -p 5000:5000 kdd-ids-app
# CPU_LIMIT/MEM_LIMIT_MB just tag rows in outputs/results.csv - keep them in sync with --cpus/--memory above.
# For docker-compose, use deploy.resources.limits (cpus: "1.0", memory: 512M)
CMD ["gunicorn", "--preload", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", "app:app"]