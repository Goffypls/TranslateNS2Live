"""Hilo de procesamiento del lado cliente: le manda frames al servidor Docker
por HTTP y guarda la última respuesta para que el render la dibuje.

Reemplaza al viejo `ProcessingThread` local: acá no corre ningún modelo de
ML, solo empaqueta el frame como JPEG, lo manda a `POST /translate` y
parsea la respuesta. El servidor es quien mantiene el tracker/caché de
traducciones entre requests (por eso no hace falta duplicar esa lógica acá).
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import requests

from ..config import AppConfig
from ..protocol import boxes_from_json
from ..types import TranslatedBox


class RemoteProcessingThread:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._session = requests.Session()

        self._latest_boxes: list[TranslatedBox] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._get_frame = None

    def start(self, get_frame) -> None:
        self._get_frame = get_frame
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        interval = self._config.pipeline.process_every_ms / 1000.0
        url = f"{self._config.server.url.rstrip('/')}/translate"
        while not self._stop_event.is_set():
            start = time.monotonic()
            frame = self._get_frame() if self._get_frame else None
            if frame is not None:
                self._send_frame(url, frame)
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, interval - elapsed))

    def _send_frame(self, url: str, frame_bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
        try:
            resp = self._session.post(
                url,
                data=buf.tobytes(),
                headers={"Content-Type": "image/jpeg"},
                timeout=self._config.server.timeout_s,
            )
            resp.raise_for_status()
            boxes = boxes_from_json(resp.json()["boxes"])
        except requests.RequestException as exc:
            print(f"[remote] no se pudo contactar al servidor ({url}): {exc}")
            return
        except (KeyError, ValueError) as exc:
            print(f"[remote] respuesta inesperada del servidor: {exc}")
            return

        with self._lock:
            self._latest_boxes = boxes

    def get_latest_boxes(self) -> list[TranslatedBox]:
        with self._lock:
            return list(self._latest_boxes)

    def clear_cache(self) -> None:
        url = f"{self._config.server.url.rstrip('/')}/clear-cache"
        try:
            self._session.post(url, timeout=self._config.server.timeout_s)
        except requests.RequestException as exc:
            print(f"[remote] no se pudo limpiar la caché remota: {exc}")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
