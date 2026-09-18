"""Traducción de texto japonés a español, con backend intercambiable.

Backends:
- `argos` (default): offline, gratuito. Si no hay paquete directo JA->ES
  instalado, encadena JA->EN->ES (`pivot_lang`).
- `deepl`: online, requiere `deepl_api_key`. Mejor calidad, útil si hay
  conexión y se prioriza precisión sobre funcionar sin internet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .config import TranslationConfig


class Translator(ABC):
    @abstractmethod
    def translate(self, text: str) -> str:
        """Traduce `text` (idioma origen configurado) al idioma destino configurado."""


class ArgosTranslator(Translator):
    def __init__(self, config: TranslationConfig) -> None:
        self._config = config
        self._direct = None
        self._pivot_1 = None  # source -> pivot
        self._pivot_2 = None  # pivot -> target
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import argostranslate.translate as argos_translate

        installed = argos_translate.get_installed_languages()
        by_code = {lang.code: lang for lang in installed}

        source = by_code.get(self._config.source_lang)
        target = by_code.get(self._config.target_lang)
        pivot = by_code.get(self._config.pivot_lang)

        if source and target:
            direct = source.get_translation(target)
            if direct is not None:
                self._direct = direct

        if self._direct is None and source and pivot and target:
            self._pivot_1 = source.get_translation(pivot)
            self._pivot_2 = pivot.get_translation(target)
            if self._pivot_1 is None or self._pivot_2 is None:
                raise RuntimeError(
                    "No hay paquetes de argos-translate instalados para "
                    f"{self._config.source_lang}->{self._config.target_lang} "
                    f"ni para el pivote {self._config.pivot_lang}. "
                    "Corré `python -m translatens2live.setup_models`."
                )
        elif self._direct is None:
            raise RuntimeError(
                "No hay paquetes de argos-translate instalados. "
                "Corré `python -m translatens2live.setup_models`."
            )

        self._loaded = True

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        self._ensure_loaded()
        if self._direct is not None:
            return self._direct.translate(text)
        assert self._pivot_1 is not None and self._pivot_2 is not None
        pivoted = self._pivot_1.translate(text)
        return self._pivot_2.translate(pivoted)


class DeepLTranslator(Translator):
    def __init__(self, config: TranslationConfig) -> None:
        if not config.deepl_api_key:
            raise ValueError("translation.deepl_api_key es requerido para el backend deepl")
        self._config = config
        self._client = None

    def _ensure_loaded(self) -> None:
        if self._client is not None:
            return
        import deepl

        self._client = deepl.Translator(self._config.deepl_api_key)

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        self._ensure_loaded()
        assert self._client is not None
        result = self._client.translate_text(
            text,
            source_lang=self._config.source_lang.upper(),
            target_lang=self._config.target_lang.upper(),
        )
        return str(result)


def get_translator(config: TranslationConfig) -> Translator:
    if config.backend == "argos":
        return ArgosTranslator(config)
    if config.backend == "deepl":
        return DeepLTranslator(config)
    raise ValueError(f"Backend de traducción desconocido: {config.backend}")
