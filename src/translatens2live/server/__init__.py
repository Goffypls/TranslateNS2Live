"""Servidor: motor de traducción (detección + OCR + traducción) expuesto por HTTP.

Pensado para correr en Docker: todos los modelos de ML son gratuitos y
corren en CPU (PaddleOCR, manga-ocr, argos-translate), sin API keys ni
servicios pagos.
"""
