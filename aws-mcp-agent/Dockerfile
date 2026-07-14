FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (this layer caches unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY math_tools.py .
COPY main.py .

# Run as a non-root user (good practice; ECS/security scanners expect this)
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# Container-level liveness check hitting our /health endpoint (no Bedrock needed).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

# NOTE: no credentials in the image. Locally we pass --env-file .env;
# on ECS Fargate the TASK ROLE supplies AWS creds automatically.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
