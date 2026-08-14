# syntax=docker/dockerfile:1
FROM python:3.13-slim AS base

# uv is the project's package manager; copy the static binary rather than pip-installing it.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependency layer first so source edits do not invalidate the install.
COPY pyproject.toml uv.lock* README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra bench --no-install-project

COPY memore ./memore
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra bench

# Both services this talks to live outside the container: FalkorDB is a sibling in
# compose, Ollama is on the host (see compose.yaml for why it is not containerised).
ENV MEMORE_FALKOR_HOST=falkordb \
    MEMORE_OLLAMA_URL=http://host.docker.internal:11434

ENTRYPOINT ["memore"]
CMD ["demo"]
