# Builder stage
FROM python:3.11-slim as builder

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

# Copy necessary files
COPY . .

# Create model cache directory
RUN mkdir -p model_cache && chmod 777 model_cache

# Environment variables with defaults
ENV GITHUB_TOKEN=""
ENV OWNER_REPO="remla25-team14/model-training"
ENV ARTIFACT_ID="3143858901" 
ENV PORT=5000

# Expose the port
EXPOSE ${PORT}

# Entry point
CMD ["python", "app.py"]
