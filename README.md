# TranslateNS2Live

Traducción en vivo (japonés → español) para consolas importadas capturadas por HDMI —
pensado originalmente para una Nintendo Switch 2 japonesa leída a través de una
capturadora de video, pero funciona con cualquier fuente de video con texto en pantalla
(japonés, coreano, chino, inglés, etc. — configurable).

A diferencia de herramientas tipo [Translumo](https://github.com/Danily07/Translumo) o
Luna, que suelen traducir un área fija de pantalla como bloque, esta app:

1. **Detecta cada recuadro de texto por separado** (cuadros de diálogo, menús, tutoriales)
   usando un detector de texto en la imagen.
2. **Reconoce el texto japonés por recuadro** con un modelo especializado en tipografía de
   videojuegos/manga (mucho más preciso que un OCR genérico para estas fuentes).
3. **Traduce y cachea por recuadro**: si el mismo texto ya se tradujo antes (diálogos que
   se repiten, menús recurrentes) no se vuelve a llamar al traductor.
4. **Sigue cada recuadro entre frames** (tracking + hash de contenido) para no
   retraducir ni parpadear mientras el cuadro de texto no cambió, y para no gastar CPU/GPU
   analizando frames idénticos.
5. Dibuja la traducción **superpuesta sobre cada recuadro detectado**, en una ventana
   propia que muestra el feed de la capturadora (no se toca la consola ni la señal
   original).

## Arquitectura

```
Capturadora (HDMI → USB) ──► cv2.VideoCapture ──► CaptureThread
                                                        │ último frame
                                                        ▼
                                              ProcessingThread (cada N ms
                                              o cuando el frame cambió)
                                                        │
                              ┌─────────────────────────┼─────────────────────────┐
                              ▼                          ▼                         ▼
                       TextDetector              RegionTracker              TranslationCache
                    (encuentra bboxes)     (empareja bboxes entre       (texto→traducción,
                              │              frames, decide qué           evita re-llamadas)
                              ▼              recuadros reprocesar)
                        JapaneseOCR
                    (reconoce texto por
                       recuadro recortado)
                              │
                              ▼
                          Translator
                       (JA → ES, con
                        caché/backend
                        intercambiable)
                              │
                              ▼
                     lista de TextBox(bbox, texto_traducido)
                              │
                              ▼
                        OverlayRenderer ──► ventana (OpenCV/PIL)
                     (dibuja el frame en vivo + cajas traducidas)
```

El video se renderiza a la tasa de fps real de la capturadora; la detección/OCR/traducción
corre en un hilo aparte y solo actualiza las cajas superpuestas cuando hace falta, así la
imagen no se traba esperando al modelo.

## Componentes (`src/translatens2live/`)

- `capture.py` — hilo de lectura de la capturadora (OpenCV `VideoCapture`).
- `detection.py` — detector de regiones de texto (backend PaddleOCR `det`, intercambiable).
- `ocr.py` — reconocimiento de japonés por recuadro (backend `manga-ocr`, intercambiable).
- `translator.py` — traducción JA→ES (backend `argos-translate` offline por defecto,
  o DeepL si se configura una API key).
- `cache.py` — caché de traducciones por texto normalizado.
- `tracker.py` — matching de cajas entre frames (IoU) + hash de contenido para decidir
  qué recuadros reprocesar.
- `overlay.py` — composición del overlay (texto con acentos/kanji vía PIL) sobre el frame.
- `pipeline.py` — orquesta captura + procesamiento en hilos separados.
- `config.py` — configuración (dispositivo de captura, idiomas, backends, umbrales).
- `app.py` — punto de entrada: abre la ventana, procesa teclado (pausar, cambiar overlay, etc).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# modelo de traducción offline JA->EN y EN->ES (una sola vez)
python -m translatens2live.setup_models
```

Windows es la plataforma recomendada porque la mayoría de capturadoras USB exponen el
dispositivo como cámara UVC estándar (backend DirectShow de OpenCV).

## Uso

1. Conectá la capturadora y confirmá qué índice de dispositivo le asigna Windows/OpenCV
   (`python -m translatens2live.list_devices`).
2. Copiá `config.example.yaml` a `config.yaml` y poné ese índice en `capture.device_index`.
3. Ejecutá:

```bash
python main.py
```

Controles en la ventana:
- `q` — salir
- `o` — mostrar/ocultar overlay de traducción
- `d` — mostrar/ocultar cajas de debug (bbox + texto japonés detectado)
- `c` — limpiar caché de traducciones en memoria

## Estado del proyecto

MVP funcional con arquitectura modular. Pendiente natural de siguientes iteraciones:
- Ajustar umbrales de estabilidad/hash por juego (algunos tienen texto animado tipo
  máquina de escribir).
- Sumar un backend de traducción online opcional (DeepL/Google) detrás de la misma
  interfaz `Translator`.
- Persistir la caché de traducciones a disco por juego (útil porque los diálogos de un
  mismo juego se repiten muchísimo).
