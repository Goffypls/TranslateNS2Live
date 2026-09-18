"""Punto de entrada: abre la ventana con el feed de la capturadora + overlay en vivo."""

from __future__ import annotations

import sys

import cv2

from .capture import CaptureThread
from .config import AppConfig, load_config
from .overlay import render_overlay
from .pipeline import ProcessingThread

WINDOW_NAME = "TranslateNS2Live"


def run(config: AppConfig) -> None:
    capture = CaptureThread(config.capture)
    capture.start()

    processing = ProcessingThread(config)
    processing.start(get_frame=capture.get_latest)

    show_overlay = True
    show_debug = config.overlay.show_debug_boxes

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = capture.get_latest()
            if frame is None:
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
                continue

            display = frame
            if show_overlay:
                boxes = processing.get_latest_boxes()
                display = render_overlay(frame, boxes, config.overlay, show_debug)

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
