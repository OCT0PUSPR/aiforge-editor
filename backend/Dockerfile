# aiforge backend — FastAPI + uvicorn.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIFORGE_HOST=0.0.0.0 \
    AIFORGE_PORT=8000 \
    AIFORGE_WORKSPACE_ROOT=/workspace

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package and create the sandboxed workspace mount point.
COPY aiforge ./aiforge
RUN mkdir -p /workspace

EXPOSE 8000

# Default to the offline mock backend; override AIFORGE_BACKEND + keys to use a
# real provider.
ENV AIFORGE_BACKEND=mock

CMD ["uvicorn", "aiforge.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
