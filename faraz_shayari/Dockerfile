FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime code + the prebuilt index (couplets + embeddings). No raw dataset,
# no scraper, no .env — Bedrock creds come from the ECS task role at runtime.
COPY app.py generate.py retrieve.py ./
COPY frontend.html ./
COPY data/couplets.jsonl data/embeddings.npy ./data/

RUN useradd -m appuser
USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
