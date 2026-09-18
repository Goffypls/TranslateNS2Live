"""Composición del overlay de traducción sobre un frame de video.

Usa PIL para dibujar el texto porque `cv2.putText` no soporta bien acentos
ni tipografías con fallback amplio de Unicode (kanji + español con tildes en
la misma pasada de debug). Se compone un fondo semitransparente detrás de
cada línea para que el texto se lea sobre cualquier imagen de fondo.
"""

from __future__ import annotations

import textwrap
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import OverlayConfig
from ..types import TranslatedBox


@lru_cache(maxsize=4)
def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    path = Path(font_path)
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _wrap_to_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int
) -> list[str]:
    if not text:
        return []
    # Aproxima el ancho de caracter para elegir cuántos caracteres por línea entran
    # en `max_width`, después reflowea con textwrap. Suficiente para texto corto de
    # diálogos de juego; no busca ser un layout engine tipográfico completo.
    bbox = draw.textbbox((0, 0), text, font=font)
    avg_char_width = max(1, (bbox[2] - bbox[0]) / max(1, len(text)))
    chars_per_line = max(1, int(max_width / avg_char_width))
    return textwrap.wrap(text, width=chars_per_line) or [text]


def render_overlay(
    frame_bgr: np.ndarray,
    boxes: list[TranslatedBox],
    config: OverlayConfig,
    show_debug: bool = False,
) -> np.ndarray:
    """Devuelve una copia de `frame_bgr` con las traducciones dibujadas encima."""

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    font = _load_font(config.font_path, config.font_size)

    bg_alpha = int(255 * config.background_opacity)
    bg_color = (*config.background_color, bg_alpha)
    text_color = (*config.text_color, 255)

    for box in boxes:
        if not box.translated_text:
            continue
        lines = _wrap_to_width(draw, box.translated_text, font, box.bbox.width or 200)

        line_heights = [
            draw.textbbox((0, 0), line, font=font)[3]
            - draw.textbbox((0, 0), line, font=font)[1]
            for line in lines
        ]
        total_height = sum(line_heights) + 4 * len(lines)

        # El overlay se dibuja pegado arriba del recuadro original (como pidió el
        # usuario: "arriba de cada cuadradito"), y si no entra por estar cerca del
        # borde superior, se dibuja adentro del propio recuadro.
        top = box.bbox.y1 - total_height - 6
        if top < 0:
            top = box.bbox.y1 + 2

        draw.rectangle(
            [box.bbox.x1, top, box.bbox.x1 + box.bbox.width, top + total_height],
            fill=bg_color,
        )

        y = top + 2
        for line, h in zip(lines, line_heights):
            draw.text((box.bbox.x1 + 4, y), line, font=font, fill=text_color)
            y += h + 4

        if show_debug:
            draw.rectangle(box.bbox.as_tuple(), outline=(0, 255, 0, 255), width=2)

    result_rgb = np.array(image)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
