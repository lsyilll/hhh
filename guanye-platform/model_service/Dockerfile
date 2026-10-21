FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml .
COPY model_service model_service
COPY worker worker
RUN pip install --no-cache-dir '.[experiments]' 'transformers>=5.2,<6' accelerate faiss-cpu timm einops iopath ftfy regex huggingface-hub
ARG INSTALL_SAM3=0
RUN if [ "$INSTALL_SAM3" = "1" ]; then pip install --no-cache-dir 'git+https://github.com/facebookresearch/sam3.git@7f473cfd983d5eb43040ff608f66bff0d7bdcb53'; fi
COPY model_service model_service
COPY worker worker
COPY integrations/telemetry.py telemetry.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
USER 10001
CMD ["uvicorn","model_service.server:app","--host","0.0.0.0","--port","8082"]
