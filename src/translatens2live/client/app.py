"""Punto de entrada del cliente: abre la ventana con el feed de la capturadora
+ overlay en vivo, traduciendo vía el servidor Docker (ver server/main.py)."""

from __future__ import annotations

import sys

import cv2

from ..config import AppConfig, load_config
from ..types import BBox
from .capture import CaptureThread
from .overlay import render_overlay
from .remote_pipeline import RemoteProcessingThread

WINDOW_NAME = "GoofypTrans"


class _RoiSelector:
    """Maneja el arrastre de mouse para elegir qué región de la pantalla traducir."""

    def __init__(self) -> None:
        self.active = False  # true mientras se está arrastrando
        self.start: tuple[int, int] | None = None
        self.current: tuple[int, int] | None = None

    def on_mouse(self, event: int, x: int, y: int, flags: int, userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.active = True
            self.start = (x, y)
            self.current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.active:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.active:
            self.active = False
            self.current = (x, y)

    def preview_bbox(self) -> BBox | None:
        if self.start is None or self.current is None:
            return None
        (x0, y0), (x1, y1) = self.start, self.current
        return BBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def consume_finished_selection(self) -> BBox | None:
        """Devuelve la región recién elegida (si el usuario terminó de arrastrar) y limpia el estado."""

        if self.active or self.start is None or self.current is None:
            return None
        bbox = self.preview_bbox()
        self.start = None
        self.current = None
        if bbox is None or bbox.width < 10 or bbox.height < 10:
            return None  # arrastre insignificante (un click sin mover), se ignora
        return bbox


def run(config: AppConfig) -> None:
    capture = CaptureThread(config.capture)
    capture.start()

    processing = RemoteProcessingThread(config)
    processing.start(get_frame=capture.get_latest)

    show_overlay = True
    show_debug = config.overlay.show_debug_boxes
    selector = _RoiSelector()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, selector.on_mouse)
    try:
        while True:
            frame = capture.get_latest()
            if frame is None:
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
                continue

            finished = selector.consume_finished_selection()
            if finished is not None:
                processing.set_roi(finished)

            display = frame
            if show_overlay:
                boxes = processing.get_latest_boxes()
                display = render_overlay(frame, boxes, config.overlay, show_debug)

            roi = processing.get_roi()
            if roi is not None:
                cv2.rectangle(display, (roi.x1, roi.y1), (roi.x2, roi.y2), (0, 200, 255), 2)
            preview = selector.preview_bbox() if selector.active else None
            if preview is not None:
                cv2.rectangle(
                    display, (preview.x1, preview.y1), (preview.x2, preview.y2), (0, 200, 255), 1
                )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("o"):
                show_overlay = not show_overlay
            elif key == ord("d"):
                show_debug = not show_debug
            elif key == ord("c"):
                processing.clear_cache()
            elif key == ord("x"):
                processing.set_roi(None)  # volver a analizar la pantalla completa
    finally:
        processing.stop()
        capture.stop()
        cv2.destroyAllWindows()


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)
    run(config)


if __name__ == "__main__":
    main()
