"""Lista los índices de dispositivo de captura de video disponibles.

Uso: python -m translatens2live.list_devices
"""

from __future__ import annotations

import cv2


def main(max_index: int = 10) -> None:
    print("Buscando dispositivos de captura (esto puede tardar unos segundos)...")
    found = False
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ok, frame = cap.read()
            shape = frame.shape if ok and frame is not None else None
            print(f"  índice {index}: disponible" + (f", frame={shape}" if shape else ""))
            found = True
        cap.release()
    if not found:
        print("  No se encontró ningún dispositivo de captura.")


if __name__ == "__main__":
    main()
