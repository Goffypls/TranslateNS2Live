"""Carga de configuración desde YAML con defaults razonables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CaptureConfig:
    device_index: int = 0
    backend: str = "dshow"
    width: int = 1920
    height: int = 1080
    fps: int = 30


@dataclass
class DetectionConfig:
    backend: str = "paddleocr"
    # Filtra recuadros chicos (etiquetas de íconos, tooltips de botones) que
    # no son diálogo/menú real. Si algún menú legítimo usa texto muy chico y
    # queda afuera, hay que bajar este valor para ese juego en particular.
    min_box_area: int = 600
    # det_db_thresh: qué tan "seguro" tiene que estar el modelo de que un
    # píxel es texto antes de agruparlo en una caja. Bajarlo agarra texto más
    # fino/liviano (títulos de menú), pero puede traer más ruido.
    det_db_thresh: float = 0.2
    # det_db_box_thresh: una vez agrupados los píxeles en una caja candidata,
    # qué tan segura tiene que estar esa caja completa para conservarla.
    det_db_box_thresh: float = 0.4
    # Cuánto se expande cada caja detectada antes de agruparla con vecinas.
    # Subirlo ayuda a que títulos con caracteres separados (típico de logos
    # de juego) se detecten como una sola caja en vez de fragmentos chicos
    # que después se descartan como ruido.
    det_db_unclip_ratio: float = 1.8
    # Junta en un solo recuadro los fragmentos que el detector separó pero
    # están en la misma línea de texto (mismo alto, cerca horizontalmente).
    # Evita traducir pedacitos de una misma oración por separado. 0 desactiva.
    line_merge_gap_factor: float = 1.5


@dataclass
class OcrConfig:
    backend: str = "manga_ocr"
    source_lang: str = "ja"


@dataclass
class TranslationConfig:
    backend: str = "argos"
    source_lang: str = "ja"
    target_lang: str = "es"
    deepl_api_key: str | None = None
    pivot_lang: str = "en"


@dataclass
class TrackerConfig:
    iou_match_threshold: float = 0.4
    content_change_threshold: int = 10
    stable_frames_to_lock: int = 2


@dataclass
class PipelineConfig:
    process_every_ms: int = 100
    max_boxes_per_frame: int = 24
    # El cliente achica el frame a este ancho máximo antes de mandarlo al
    # servidor (0 = mandar a resolución original). PaddleOCR/manga-ocr
    # tardan más cuanto más grande es la imagen, así que esto es la palanca
    # más grande para bajar la latencia end-to-end.
    max_send_width: int = 1280
    # Si la última traducción recibida para una caja es más vieja que esto,
    # se deja de dibujar en vez de quedar "pegada" mostrando algo que ya no
    # corresponde a lo que se ve en pantalla.
    overlay_max_age_s: float = 1.0
    # El servidor compara cada frame contra el anterior (en miniatura, en
    # escala de grises); si la diferencia promedio de brillo por píxel no
    # supera esto, se considera "la misma pantalla" y NO se corre detección
    # ni OCR ni traducción: se devuelve la traducción ya calculada, sin
    # tocarla. Esto es lo que evita que el overlay tiemble/titile en una
    # pantalla estática. Subilo si notás que tarda en reaccionar a cambios
    # reales; bajalo si tarda en "congelarse" del todo.
    frame_change_threshold: float = 2.5


@dataclass
class OverlayConfig:
    font_path: str = "assets/fonts/NotoSansJP-Regular.otf"
    font_size: int = 28
    min_font_size: int = 18        # no reduce la letra de la traducción por debajo de esto
    corner_radius: int = 10        # esquinas redondeadas del fondo, en px
    background_opacity: float = 0.75
    text_color: tuple[int, int, int] = (255, 255, 255)
    background_color: tuple[int, int, int] = (10, 10, 10)
    show_debug_boxes: bool = False


@dataclass
class CacheConfig:
    persist_path: str | None = None


@dataclass
class ServerConfig:
    """Solo la usa el cliente: dónde encontrar el servidor Docker con el motor de traducción."""

    host: str = "0.0.0.0"
    port: int = 8000
    url: str = "http://localhost:8000"
    timeout_s: float = 15.0


@dataclass
class AppConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


_SECTION_TYPES = {
    "capture": CaptureConfig,
    "detection": DetectionConfig,
    "ocr": OcrConfig,
    "translation": TranslationConfig,
    "tracker": TrackerConfig,
    "pipeline": PipelineConfig,
    "overlay": OverlayConfig,
    "cache": CacheConfig,
    "server": ServerConfig,
}


def _build_section(section_cls: type, data: dict[str, Any]) -> Any:
    valid_fields = {f for f in section_cls.__dataclass_fields__}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return section_cls(**filtered)


def load_config(path: str | Path | None) -> AppConfig:
    """Carga `path` (YAML) sobre los defaults. Si `path` no existe, devuelve defaults."""

    raw: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}

    kwargs: dict[str, Any] = {}
    for name, cls in _SECTION_TYPES.items():
        section_data = raw.get(name, {}) or {}
        kwargs[name] = _build_section(cls, section_data)
    return AppConfig(**kwargs)
