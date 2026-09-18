"""Servidor HTTP del motor de traducción. Pensado para correr dentro de Docker.

Uso local (sin Docker): uvicorn translatens2live.server.main:app --port 8000
Dentro de Docker: ver Dockerfile / docker-compose.yml en la raíz del repo.

CONFIG_PATH (env var) apunta al config.yaml a usar (default: /app/config.yaml
dentro del contenedor, config.yaml en la raíz si se corre local).
"""

from __future__ import annotations

import os
import threading

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..config import load_config
from ..processor import FrameProcessor
from ..protocol import boxes_to_json

_config_path = os.environ.get("CONFIG_PATH", "config.yaml")
_config = load_config(_config_path)

app = FastAPI(title="TranslateNS2Live engine")
_processor = FrameProcessor(_config)
_lock = threading.Lock()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "cache_size": _processor.cache_size()}


@app.post("/translate")
async def translate(request: Request) -> JSONResponse:
    """Recibe un frame como JPEG crudo en el body y devuelve los recuadros traducidos."""

    body = await request.body()
    if not body:
        return JSONResponse({"error": "empty body"}, status_code=400)

    array = np.frombuffer(body, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    # Los modelos (PaddleOCR/manga-ocr) no son necesariamente thread-safe y el
    # tracker/caché son estado compartido: se serializan los requests. Para un
    # único jugador viendo un único stream esto no es un cuello de botella
    # real (el cliente ya throttlea el envío con `pipeline.process_every_ms`).
    with _lock:
        boxes = _processor.process(frame)

    return JSONResponse({"boxes": boxes_to_json(boxes)})


@app.post("/clear-cache")
async def clear_cache() -> dict:
    with _lock:
        _processor.clear_cache()
    return {"ok": True}


@app.on_event("shutdown")
def _on_shutdown() -> None:
    _processor.save_cache()
