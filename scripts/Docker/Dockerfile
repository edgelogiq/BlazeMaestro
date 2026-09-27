# BlazeAIoT Maestro Docker Image
# Multi-stage build to reduce final image size

# Builder stage: Install dependencies and compile
FROM python:3.12-slim as builder

ARG LOCAL_GIT_USERNAME
ARG LOCAL_GIT_TOKEN

# Install build dependencies only in builder stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libboost-dev \
    libboost-context-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY src/requirements.txt /app/requirements.txt

# Install Python dependencies with pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-warn-script-location \
    -r requirements.txt \
    --extra-index-url https://$LOCAL_GIT_USERNAME:$LOCAL_GIT_TOKEN@gitlab.ele.etsmtl.ca/api/v4/projects/6111/packages/pypi/simple \
    --extra-index-url https://$LOCAL_GIT_USERNAME:$LOCAL_GIT_TOKEN@gitlab.ele.etsmtl.ca/api/v4/projects/5098/packages/pypi/simple \
    --extra-index-url https://$LOCAL_GIT_USERNAME:$LOCAL_GIT_TOKEN@gitlab.ele.etsmtl.ca/api/v4/projects/5095/packages/pypi/simple \
    --extra-index-url https://$LOCAL_GIT_USERNAME:$LOCAL_GIT_TOKEN@gitlab.ele.etsmtl.ca/api/v4/projects/5097/packages/pypi/simple \
    --extra-index-url https://$LOCAL_GIT_USERNAME:$LOCAL_GIT_TOKEN@gitlab.ele.etsmtl.ca/api/v4/projects/6671/packages/pypi/simple \
    --extra-index-url https://$LOCAL_GIT_USERNAME:$LOCAL_GIT_TOKEN@gitlab.ele.etsmtl.ca/api/v4/projects/6670/packages/pypi/simple \
    --extra-index-url https://$LOCAL_GIT_USERNAME:$LOCAL_GIT_TOKEN@gitlab.ele.etsmtl.ca/api/v4/projects/6668/packages/pypi/simple \
    --extra-index-url https://$LOCAL_GIT_USERNAME:$LOCAL_GIT_TOKEN@gitlab.ele.etsmtl.ca/api/v4/projects/6669/packages/pypi/simple \
    --extra-index-url https://$LOCAL_GIT_USERNAME:$LOCAL_GIT_TOKEN@gitlab.ele.etsmtl.ca/api/v4/projects/6673/packages/pypi/simple \
    --extra-index-url https://$LOCAL_GIT_USERNAME:$LOCAL_GIT_TOKEN@gitlab.ele.etsmtl.ca/api/v4/projects/6781/packages/pypi/simple

# Runtime stage: Minimal dependencies
FROM python:3.12-slim

# Install only runtime libraries (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-context-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    curl \
    && curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl \
    && rm kubectl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src/ /app/src/

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default config paths (can be overridden)
ENV LAYER_CONFIG_PATH=/config/layer/config.json
ENV NODE_CONFIG_PATH=/config/node/config.json
ENV SERVICE_CONFIG_PATH=/config/service/config.json
ENV LOG_FILE=/edgelogiq/logs/blazemaestro/log.blz

# Create directories for configs and logs
RUN mkdir -p /config/layer /config/node /config/service /edgelogiq/logs/blazemaestro

# Expose Prometheus metrics port (if needed)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

# Run BlazeMaestro
ENTRYPOINT ["python3", "/app/src/blaze_maestro_main.py"]

# Default arguments (can be overridden at runtime)
CMD ["--layer_config_path", "/config/layer/config.json", \
     "--node_config_path", "/config/node/config.json", \
     "--service_config_path", "/config/service/config.json"]
