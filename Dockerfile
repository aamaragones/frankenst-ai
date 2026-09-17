FROM python:3.12.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -fsSL https://astral.sh/uv/install.sh | sh && \
    apt-get purge -y --auto-remove curl && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/app/.venv/bin:/root/.local/bin:$PATH"

COPY pyproject.toml uv.lock README-pypi.md LICENSE /app/
COPY src /app/src
RUN uv sync --frozen --no-dev --extra examples

EXPOSE 8000

CMD ["python", "src/services/mcp/server_oaklang_agent.py"]
