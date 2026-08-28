# Ringfence — single image, two entrypoints (API and dashboard).
#
# One image rather than two because the dashboard's offline fallback path
# imports the same pipeline modules the API does; splitting them would mean
# maintaining two dependency sets that must not drift.

FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    RINGFENCE_JSON_LOGS=1

# libgomp1 is LightGBM's OpenMP runtime; without it `import lightgbm` fails at
# load time with an unhelpful shared-object error.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a source edit does not invalidate the install layer.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY frontend/ ./frontend/
COPY tests/ ./tests/
COPY pytest.ini Makefile README.md ARCHITECTURE.md BARS.md ./
COPY notebooks/ ./notebooks/
COPY demo/ ./demo/

RUN mkdir -p data/raw data/processed data/ground_truth

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
