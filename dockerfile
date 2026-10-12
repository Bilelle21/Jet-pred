FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app


COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.11-slim

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /app /app
COPY build/model /app/model

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_DIR=/app/model

USER 10001
EXPOSE 8000

CMD ["uvicorn", "jetpred.serve:app", "--host", "0.0.0.0", "--port", "8000"]