FROM python:3.12-slim

WORKDIR /app

# System deps for pyarrow/torch wheels and health checks.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first, separately from app code, so a code-only
# change doesn't invalidate the (large - torch) dependency layer's cache.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# mlruns (MLflow's local file store) and data are mounted at runtime via
# docker-compose volumes rather than baked into the image, since they're
# generated/updated outside of Docker (from your own training runs).
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "f1telemetry.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
