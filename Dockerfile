# SQL Repair Environment
# Build:  docker build -t sql-repair-env .
# Run:    docker run -p 7860:7860 -e HF_TOKEN="AIza..." sql-repair-env
# Verify: curl http://localhost:7860/health

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer cached)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project (build context = repo root)
COPY . .

# Hackathon-required env vars — override at runtime with -e
ENV PORT=7860
ENV API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
ENV MODEL_NAME="gemini-2.5-flash"
ENV HF_TOKEN=""
ENV BASE_URL="http://localhost:7860"
ENV PYTHONPATH="/app"

EXPOSE 7860

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]