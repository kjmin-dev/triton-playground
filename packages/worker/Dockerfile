FROM python:3.10.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY packages/worker/pyproject.toml .
RUN uv pip install --system -r pyproject.toml

COPY packages/worker/ .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
