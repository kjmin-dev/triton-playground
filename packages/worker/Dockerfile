FROM python:3.10.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY packages/worker/pyproject.toml ./pyproject.toml
COPY packages/worker/api ./api
COPY packages/worker/pipeline ./pipeline
RUN uv pip install --system .

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${WORKER_PORT:-8080}"]
