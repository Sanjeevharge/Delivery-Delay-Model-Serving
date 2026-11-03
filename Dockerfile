FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY u2/ u2/
COPY artifacts/model_bundle.joblib artifacts/model_bundle.joblib
ENV MODEL_PATH=/app/artifacts/model_bundle.joblib PRED_LOG=/app/logs/predictions.jsonl
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "u2.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
