# STAGE 1: builder
FROM python:3.12-slim AS builder
WORKDIR /app

# Install build deps (gcc for wheels, git for VCS deps)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      gcc \
      git \
 && rm -rf /var/lib/apt/lists/*

# Copy & install Python deps (including libml from GitHub)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt

# Copy your code (app.py, libml/, etc.)
COPY . .

# STAGE 2: runtime
FROM python:3.12-slim
WORKDIR /app

# Bring in installed packages
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

ENV PORT=5000
EXPOSE 5000

CMD ["python", "app.py"]
