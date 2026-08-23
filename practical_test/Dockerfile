# Build from project root:
#   docker build -f practical_test/Dockerfile -t xai-ids-server .
# Run:
#   docker run -p 8080:8080 xai-ids-server

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY practical_test/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY practical_test/app.py /app/app.py

# Trained models
COPY models/rf_model_KDDCup99_10k.joblib  /app/models/rf_model_KDDCup99_10k.joblib
COPY models/rf_model_UNSW-NB15_10k.joblib /app/models/rf_model_UNSW-NB15_10k.joblib

# Log directory
RUN mkdir -p /app/logs

EXPOSE 8080

ENV PORT=8080 \
    MODELS_DIR=/app/models \
    LOG_DIR=/app/logs

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
