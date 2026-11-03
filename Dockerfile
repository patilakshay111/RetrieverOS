FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY data ./data
COPY alembic.ini ./alembic.ini
COPY migrations ./migrations
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.15.0+cpu" \
    && pip install . \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/runtime \
    && chown -R appuser:appuser /app

USER 10001:10001

EXPOSE 8000
