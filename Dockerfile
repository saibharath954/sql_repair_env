# SQL Repair Environment
# Build:  docker build -t sql-repair-env .
# Run:    docker run -p 7860:7860 -e HF_TOKEN="your_token" sql-repair-env

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer cached)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project (build context = repo root)
COPY . .

# Hackathon-required env vars
ENV PORT=7860
ENV BASE_URL="http://localhost:7860"
ENV PYTHONPATH="/app"

EXPOSE 7860

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]