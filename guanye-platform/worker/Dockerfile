FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY worker worker
COPY model_service model_service
COPY integrations/telemetry.py telemetry.py
RUN pip install --no-cache-dir '.[experiments]' opencv-python-headless kafka-python-ng
USER 10001
ENV PYTHONUNBUFFERED=1
CMD ["celery","-A","worker.jobs","worker","--concurrency=1","--loglevel=INFO"]
