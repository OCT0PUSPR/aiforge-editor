# aiforge backend — multi-stage, non-root, with a HEALTHCHECK.
# Stage 1: build wheels for the runtime deps.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# Stage 2: slim runtime image.
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIFORGE_HOST=0.0.0.0 \
    AIFORGE_PORT=8000 \
    AIFORGE_BACKEND=mock \
    AIFORGE_DATA_DIR=/data

# Non-root user.
RUN groupadd --gid 10001 aiforge \
    && useradd --uid 10001 --gid aiforge --create-home aiforge

WORKDIR /app

# Install from prebuilt wheels (no build toolchain in the final image).
COPY --from=builder /wheels /wheels
COPY requirements.txt ./
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# App code + migrations.
COPY aiforge ./aiforge
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini

# Writable data dir owned by the non-root user.
RUN mkdir -p /data && chown -R aiforge:aiforge /data /app
USER aiforge

EXPOSE 8000

# Liveness via the /health endpoint.
HEALTHCHECK --interval=30s --timeout=4s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

# Apply migrations, then serve.
CMD ["sh", "-c", "alembic upgrade head && uvicorn aiforge.api.server:app --host 0.0.0.0 --port 8000"]
