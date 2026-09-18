"""Caché de traducciones por texto normalizado.

Los diálogos de un juego se repiten mucho (mismo NPC, mismos ítems, mismas frases
de menú), así que evitar volver a llamar al traductor por texto ya visto es la
optimización más simple y con más impacto en latencia/costo.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    """Normaliza texto japonés para usarlo como clave de caché.

    Colapsa espacios/whitespace de ancho completo y normaliza a NFKC para que
    variantes triviales (medio-ancho vs ancho-completo, espacios extra que a
    veces mete el OCR) compartan la misma entrada de caché.
    """

    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


class TranslationCache:
    """Caché en memoria, con persistencia opcional a un JSON en disco."""

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._persist_path = Path(persist_path) if persist_path else None
        self._store: dict[str, str] = {}
        if self._persist_path and self._persist_path.exists():
            self._store = json.loads(self._persist_path.read_text(encoding="utf-8"))

    def get(self, source_text: str) -> str | None:
        return self._store.get(normalize_text(source_text))

    def put(self, source_text: str, translated_text: str) -> None:
        self._store[normalize_text(source_text)] = translated_text

    def __contains__(self, source_text: str) -> bool:
        return normalize_text(source_text) in self._store

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()

    def save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(
            json.dumps(self._store, ensure_ascii=False, indent=2), encoding="utf-8"
        )
