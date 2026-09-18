"""Composición del overlay de traducción sobre un frame de video.

Usa PIL para dibujar el texto porque `cv2.putText` no soporta bien acentos
ni tipografías con fallback amplio de Unicode (kanji + español con tildes en
la misma pasada). El overlay se dibuja directamente sobre el recuadro
original (tapando el texto japonés) en vez de flotar arriba, con el tamaño
de letra ajustado automáticamente para que la traducción entre en la caja
sin desbordar ni quedar microscópica.
"""

from __future__ import annotations

import textwrap
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import OverlayConfig
from ..types import BBox, TranslatedBox

_PADDING = 7
_LINE_SPACING = 4


@lru_cache(maxsize=16)
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


def _measure_block(
    draw: ImageDraw.ImageDraw, lines: list[str], font: ImageFont.FreeTypeFont
) -> tuple[int, int]:
    if not lines:
        return 0, 0
    widths = []
    heights = []
    for line in lines:
        l, t, r, b = draw.textbbox((0, 0), line, font=font)
        widths.append(r - l)
        heights.append(b - t)
    block_w = max(widths)
    block_h = sum(heights) + _LINE_SPACING * (len(lines) - 1)
    return block_w, block_h


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str,
    box_w: int,
    box_h: int,
    max_size: int,
    min_size: int,
) -> tuple[ImageFont.FreeTypeFont, list[str], int, int]:
    """Busca, de mayor a menor, el tamaño de letra más grande que entra en la caja."""

    available_w = max(1, box_w - 2 * _PADDING)
    available_h = max(1, box_h - 2 * _PADDING)

    last: tuple[ImageFont.FreeTypeFont, list[str], int, int] | None = None
    for size in range(max_size, min_size - 1, -1):
        font = _load_font(font_path, size)
        lines = _wrap_to_width(draw, text, font, available_w)
        block_w, block_h = _measure_block(draw, lines, font)
        last = (font, lines, block_w, block_h)
        if block_h <= available_h and block_w <= available_w:
            return last

    assert last is not None
    return last


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

    bg_alpha = int(255 * config.background_opacity)
    bg_color = (*config.background_color, bg_alpha)
    text_color = (*config.text_color, 255)
    stroke_color = (0, 0, 0, min(255, bg_alpha + 40))
    frame_h, frame_w = frame_bgr.shape[:2]

    for box in boxes:
        if not box.translated_text:
            continue

        max_size = max(config.font_size, config.min_font_size)
        font, lines, block_w, block_h = _fit_text(
            draw,
            box.translated_text,
            config.font_path,
            box.bbox.width,
            box.bbox.height,
            max_size,
            config.min_font_size,
        )
        if not lines:
            continue

        # El fondo cubre como mínimo el recuadro original (tapa el texto en
        # japonés); si la traducción no entra ni al tamaño mínimo, la caja
        # crece un poco en vez de recortar el texto, manteniéndose centrada
        # en el mismo lugar.
        rect_w = max(box.bbox.width, block_w + 2 * _PADDING)
        rect_h = max(box.bbox.height, block_h + 2 * _PADDING)
        cx = box.bbox.x1 + box.bbox.width / 2
        cy = box.bbox.y1 + box.bbox.height / 2
        rect_x1 = max(0, min(frame_w - rect_w, cx - rect_w / 2))
        rect_y1 = max(0, min(frame_h - rect_h, cy - rect_h / 2))
        rect_x2 = rect_x1 + rect_w
        rect_y2 = rect_y1 + rect_h

        radius = min(config.corner_radius, int(rect_w // 2), int(rect_h // 2))
        draw.rounded_rectangle(
            [rect_x1, rect_y1, rect_x2, rect_y2], radius=max(0, radius), fill=bg_color
        )

        stroke_width = max(1, round(getattr(font, "size", config.font_size) / 16))
        y = rect_y1 + (rect_h - block_h) / 2
        for line in lines:
            l, t, r, b = draw.textbbox((0, 0), line, font=font)
            line_w = r - l
            x = rect_x1 + (rect_w - line_w) / 2
            draw.text(
                (x, y - t),
                line,
                font=font,
                fill=text_color,
                stroke_width=stroke_width,
                stroke_fill=stroke_color,
            )
            y += (b - t) + _LINE_SPACING

        if show_debug:
            draw.rectangle(box.bbox.as_tuple(), outline=(0, 255, 0, 255), width=2)

    result_rgb = np.array(image)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
