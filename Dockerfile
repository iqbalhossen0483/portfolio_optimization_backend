FROM python:3.11-slim

WORKDIR /app

# System dependencies for PyTorch + asyncpg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/

# Model store directory
RUN mkdir -p /app/model_store

# Non-root user for security
RUN useradd -m -u 1000 madrl && chown -R madrl:madrl /app
USER madrl

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
