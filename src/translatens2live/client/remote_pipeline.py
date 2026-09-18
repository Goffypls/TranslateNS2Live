"""Hilo de procesamiento del lado cliente: le manda frames al servidor Docker
por HTTP y guarda la última respuesta para que el render la dibuje.

Reemplaza al viejo `ProcessingThread` local: acá no corre ningún modelo de
ML, solo empaqueta el frame como JPEG, lo manda a `POST /translate` y
parsea la respuesta. El servidor es quien mantiene el tracker/caché de
traducciones entre requests (por eso no hace falta duplicar esa lógica acá).

Dos cosas para que se sienta rápido y no "pegado":
- El frame se achica antes de mandarlo (`pipeline.max_send_width`): PaddleOCR
  y manga-ocr tardan más cuanto más grande es la imagen, así que reducir la
  resolución de envío es la palanca más grande contra la latencia. Las
  cajas que devuelve el servidor están en esas coordenadas achicadas, así
  que se reescalan de vuelta al tamaño real antes de dibujarlas.
- Si la última respuesta exitosa es más vieja que `pipeline.overlay_max_age_s`
  (el servidor está lento, se cayó una request, etc.), se deja de mostrar en
  vez de quedar una traducción vieja pegada sobre contenido que ya cambió.
- Si un frame puntual no devuelve cajas (el detector no encontró nada esta
  vez, algo normal en video en vivo), NO se borra el overlay de inmediato:
  se mantiene la última traducción no vacía durante `overlay_max_age_s`
  antes de desaparecer. Sin esto, el overlay titilaba cada vez que el
  detector fallaba un frame suelto.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import requests

from ..config import AppConfig
from ..protocol import boxes_from_json
from ..types import BBox, TranslatedBox


def _clamp_roi(roi: BBox, frame_w: int, frame_h: int) -> BBox | None:
    x1 = max(0, min(roi.x1, frame_w - 1))
    y1 = max(0, min(roi.y1, frame_h - 1))
    x2 = max(x1 + 1, min(roi.x2, frame_w))
    y2 = max(y1 + 1, min(roi.y2, frame_h))
    return BBox(x1, y1, x2, y2)


def _scale_box(box: TranslatedBox, factor: float) -> TranslatedBox:
    b = box.bbox
    return TranslatedBox(
        bbox=BBox(
            round(b.x1 * factor), round(b.y1 * factor), round(b.x2 * factor), round(b.y2 * factor)
        ),
        source_text=box.source_text,
        translated_text=box.translated_text,
        stable=box.stable,
    )


def _offset_box(box: TranslatedBox, dx: int, dy: int) -> TranslatedBox:
    b = box.bbox
    return TranslatedBox(
        bbox=BBox(b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy),
        source_text=box.source_text,
        translated_text=box.translated_text,
        stable=box.stable,
    )


class RemoteProcessingThread:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._session = requests.Session()

        self._latest_boxes: list[TranslatedBox] = []
        self._last_contact_time = 0.0   # última vez que el servidor respondió (para detectar caídas)
        self._last_nonempty_time = 0.0  # última vez que esa respuesta trajo cajas
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._get_frame = None
        self._roi: BBox | None = None  # región elegida por el usuario; None = pantalla completa

    def set_roi(self, roi: BBox | None) -> None:
        """Restringe el análisis a esta región del frame (None = volver a full frame)."""

        self._roi = roi

    def get_roi(self) -> BBox | None:
        return self._roi

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
        roi_offset_x = roi_offset_y = 0
        roi = self._roi
        if roi is not None:
            full_h, full_w = frame_bgr.shape[:2]
            clamped = _clamp_roi(roi, full_w, full_h)
            if clamped is None:
                return
            frame_bgr = frame_bgr[clamped.y1 : clamped.y2, clamped.x1 : clamped.x2]
            roi_offset_x, roi_offset_y = clamped.x1, clamped.y1
            if frame_bgr.size == 0:
                return

        frame_h, frame_w = frame_bgr.shape[:2]
        max_width = self._config.pipeline.max_send_width
        scale = 1.0
        to_send = frame_bgr
        if max_width and frame_w > max_width:
            scale = max_width / frame_w
            to_send = cv2.resize(
                frame_bgr, (max_width, round(frame_h * scale)), interpolation=cv2.INTER_AREA
            )

        ok, buf = cv2.imencode(".jpg", to_send, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
        try:
            resp = self._session.post(
                url,
                data=buf.tobytes(),
                headers={"Content-Type": "image/jpeg"},
                timeout=self._config.server.timeout_s,
            )
            if resp.status_code == 503:
                print("[remote] el servidor todavía está cargando los modelos, esperando...")
                return
            resp.raise_for_status()
            boxes = boxes_from_json(resp.json()["boxes"])
        except requests.RequestException as exc:
            print(f"[remote] no se pudo contactar al servidor ({url}): {exc}")
            return
        except (KeyError, ValueError) as exc:
            print(f"[remote] respuesta inesperada del servidor: {exc}")
            return

        if scale != 1.0:
            inv = 1.0 / scale
            boxes = [_scale_box(b, inv) for b in boxes]
        if roi_offset_x or roi_offset_y:
            boxes = [_offset_box(b, roi_offset_x, roi_offset_y) for b in boxes]

        max_age = self._config.pipeline.overlay_max_age_s
        now = time.monotonic()
        with self._lock:
            self._last_contact_time = now
            if boxes:
                self._latest_boxes = boxes
                self._last_nonempty_time = now
            elif now - self._last_nonempty_time > max_age:
                # Ya pasó suficiente tiempo sin encontrar nada: recién ahí se
                # borra. Un solo frame sin cajas no alcanza para ocultar el
                # overlay (evita el titileo).
                self._latest_boxes = []

    def get_latest_boxes(self) -> list[TranslatedBox]:
        with self._lock:
            if time.monotonic() - self._last_contact_time > self._config.pipeline.overlay_max_age_s:
                return []  # el servidor no responde hace rato: no mostrar nada viejo
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
