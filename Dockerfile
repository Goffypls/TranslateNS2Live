# Motor de traducción (servidor). CPU-only, todo software libre/gratuito:
# PaddleOCR (detección de texto), manga-ocr (OCR japonés), argos-translate
# (traducción JA->ES offline). Sin API keys ni servicios pagos.

FROM python:3.11-slim

# libgl1/libglib2.0-0: dependencias runtime de opencv-python-headless y paddlepaddle.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY src ./src
COPY config.example.yaml ./config.yaml

ENV PYTHONPATH=/app/src
ENV CONFIG_PATH=/app/config.yaml

EXPOSE 8000

CMD ["uvicorn", "translatens2live.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
