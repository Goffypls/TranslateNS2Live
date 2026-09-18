"""Hilo de lectura de la capturadora de video.

Lee frames tan rápido como la capturadora los entregue y siempre deja
disponible el último frame leído (`get_latest`), para que el hilo de
render y el de procesamiento (detección+OCR+traducción, que es más lento)
puedan consumir a su propio ritmo sin acumular retraso ("solo lo último
importa" para video en vivo).
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from ..config import CaptureConfig

_BACKEND_FLAGS = {
    "dshow": cv2.CAP_DSHOW,
    "v4l2": cv2.CAP_V4L2,
    "any": cv2.CAP_ANY,
}


class CaptureThread:
    def __init__(self, config: CaptureConfig) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None
        self._latest_frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        backend = _BACKEND_FLAGS.get(self._config.backend, cv2.CAP_ANY)
        self._cap = cv2.VideoCapture(self._config.device_index, backend)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir el dispositivo de captura {self._config.device_index} "
                f"(backend={self._config.backend}). Probá `python -m translatens2live.list_devices`."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._cap is not None
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest_frame = frame

    def get_latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
