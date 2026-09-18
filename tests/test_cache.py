import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from translatens2live.cache import TranslationCache, normalize_text


def test_normalize_collapses_whitespace_and_width():
    assert normalize_text("こんにちは　世界") == "こんにちは 世界"
    assert normalize_text("  hola   mundo  ") == "hola mundo"


def test_cache_put_get_roundtrip():
    cache = TranslationCache()
    assert cache.get("こんにちは") is None
    cache.put("こんにちは", "Hola")
    assert cache.get("こんにちは") == "Hola"
    assert "こんにちは" in cache
    assert len(cache) == 1


def test_cache_normalizes_key_on_lookup():
    cache = TranslationCache()
    cache.put("こんにちは　世界", "Hola mundo")
    assert cache.get("こんにちは 世界") == "Hola mundo"


def test_cache_persist_roundtrip(tmp_path):
    path = tmp_path / "cache.json"
    cache = TranslationCache(persist_path=path)
    cache.put("ありがとう", "Gracias")
    cache.save()

    reloaded = TranslationCache(persist_path=path)
    assert reloaded.get("ありがとう") == "Gracias"


def test_cache_clear():
    cache = TranslationCache()
    cache.put("a", "b")
    cache.clear()
    assert len(cache) == 0
