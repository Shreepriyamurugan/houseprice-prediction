# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_SOURCE=joblib

WORKDIR /app

COPY pyproject.toml params.yaml ./
COPY api/ api/
COPY src/ src/
COPY models/model.joblib models/input_defaults.json models/metrics.json models/

RUN useradd -m -u 1000 app && \
    mkdir -p /app/logs && \
    chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').getcode() == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
