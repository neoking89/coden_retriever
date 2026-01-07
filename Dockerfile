# syntax=docker/dockerfile:1

# Coden-Retriever Docker Image
# Multi-stage build for optimized image size

# Build stage
FROM python:3.11-slim AS builder

# Build-time label only (OCI labels go in runtime stage)
LABEL stage="builder"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Build the wheel
RUN pip install build && python -m build --wheel

# Runtime stage
FROM python:3.11-slim AS runtime

# Image metadata (OCI standard labels)
LABEL org.opencontainers.image.title="coden-retriever"
LABEL org.opencontainers.image.description="Code analysis and search tool with MCP server support"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="Coden Retriever"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/coden-retriever/coden-retriever"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Default to HTTP transport for MCP server
    CODEN_RETRIEVER_TRANSPORT=http \
    # Bind to all interfaces for container networking
    CODEN_RETRIEVER_HOST=0.0.0.0 \
    CODEN_RETRIEVER_PORT=8000

WORKDIR /app

# Install runtime dependencies and create non-root user
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 1000 coden

# Copy the wheel from builder stage
COPY --from=builder /build/dist/*.whl /tmp/

# Install the package
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Create cache, config, and workspace directories with correct ownership
# Cache: ~/.coden-retriever (where the tool stores indices)
# Config: ~/.coden-retriever (where settings.json lives)
# Create default config with Docker-friendly Ollama URL (host.docker.internal)
RUN mkdir -p /workspace /home/coden/.coden-retriever \
    && echo '{"_version":1,"model":{"default":"ollama:","base_url":null,"provider_urls":{"ollama":"http://host.docker.internal:11434/v1","llamacpp":"http://host.docker.internal:8080/v1"}}}' \
         > /home/coden/.coden-retriever/settings.json \
     && chown -R coden:coden /workspace /home/coden/.coden-retriever

# Switch to non-root user
USER coden

# Set working directory for code analysis
WORKDIR /workspace

# Expose default MCP server port
EXPOSE 8000

# Health check for MCP server mode
# Uses Python's built-in urllib (no curl in slim images)
# The check is skipped gracefully in CLI mode (container exits before unhealthy)
# Note: Using shell form to allow environment variable expansion
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Ensure proper signal handling for graceful shutdown
STOPSIGNAL SIGTERM

# Default command: start MCP server in HTTP mode
CMD ["coden", "serve", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
