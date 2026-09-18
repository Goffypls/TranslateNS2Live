# TranslateNS2Live

Traducción en vivo (japonés → español) para consolas importadas capturadas por HDMI —
pensado originalmente para una Nintendo Switch 2 japonesa leída a través de una
capturadora de video, pero funciona con cualquier fuente de video con texto en pantalla.

A diferencia de herramientas tipo [Translumo](https://github.com/Danily07/Translumo) o
Luna, que suelen traducir un área fija de pantalla como bloque, esta app:

1. **Detecta cada recuadro de texto por separado** (cuadros de diálogo, menús, tutoriales).
2. **Reconoce el texto japonés por recuadro** con un modelo especializado en tipografía de
   videojuegos/manga.
3. **No vuelve a traducir un recuadro hasta que detecta un cambio real**: cada caja se
   sigue entre frames (posición + un hash del contenido); si sigue igual, se reusa la
   traducción ya calculada sin llamar de nuevo al OCR ni al traductor. Y si el mismo texto
   aparece en otro lado de la pantalla (diálogos que se repiten, menús recurrentes), se
   reusa desde una caché por texto en vez de retraducir.
4. Dibuja la traducción **superpuesta arriba de cada recuadro detectado**, en una ventana
   propia (no se toca la consola ni la señal original).

Todo el stack es **gratis y corre en CPU, sin API keys ni servicios pagos**: PaddleOCR
(detección), manga-ocr (OCR japonés), argos-translate (traducción JA→ES offline). Un
backend online opcional (DeepL) queda disponible por si alguna vez querés mejor calidad a
cambio de una API key propia, pero no hace falta para que la app funcione.

## Arquitectura: servidor (Docker) + cliente liviano

El proyecto se separa en dos partes que se hablan por HTTP en `localhost`:

```
┌─────────────────────────────┐        HTTP (POST /translate,        ┌───────────────────────────────┐
│   CLIENTE (tu PC, sin Docker)│        frame JPEG -> recuadros)      │  SERVIDOR (Docker)              │
│                              │ ────────────────────────────────────▶│                                 │
│  capturadora ─▶ CaptureThread│                                      │  FastAPI                        │
│                     │        │ ◀────────────────────────────────────│    │                             │
│                     ▼        │        JSON: [{bbox, texto_es}, …]   │    ▼                             │
│  RemoteProcessingThread ─────┘                                      │  FrameProcessor                 │
│   (manda 1 frame cada         (solo manda frames cuando toca,       │   ├─ TextDetector (PaddleOCR)    │
│    pipeline.process_every_ms)  el cliente nunca corre modelos)      │   ├─ JapaneseOcr (manga-ocr)     │
│                     │                                                │   ├─ RegionTracker (evita         │
│                     ▼                                                │   │   retraducir sin cambios)   │
│  OverlayRenderer ─▶ ventana                                          │   ├─ TranslationCache (evita     │
│  (dibuja el frame en vivo                                            │   │   retraducir texto repetido) │
│   a su propia tasa de fps                                            │   └─ Translator (argos-translate)│
│   + últimas cajas recibidas)                                         │                                 │
└─────────────────────────────┘                                      └───────────────────────────────┘
```

¿Por qué separar así en vez de meter todo en un solo contenedor? La ventana con el
overlay y el acceso al dispositivo USB de la capturadora necesitan correr directo en tu
PC (meter eso en Docker en Windows implica pasaje de dispositivos USB + un servidor X
para la ventana gráfica: mucho más fragil que la alternativa). En cambio los modelos de
ML (PaddleOCR, manga-ocr, argos-translate) no necesitan ni hardware especial ni GUI, así
que van en Docker: instalación reproducible, no ensucian tu Python del sistema, y podés
tirar el contenedor y levantarlo de nuevo sin reinstalar nada.

## Componentes (`src/translatens2live/`)

Compartido:
- `types.py` — `BBox`, `TranslatedBox`.
- `config.py` — configuración (un solo `config.yaml` para cliente y servidor).
- `protocol.py` — serialización JSON de `TranslatedBox` para el HTTP entre ambos.

Servidor (`server/`, corre en Docker):
- `detection.py` — detector de regiones de texto (PaddleOCR, backend intercambiable).
- `ocr.py` — reconocimiento de japonés por recuadro (manga-ocr, backend intercambiable).
- `translator.py` — traducción JA→ES (argos-translate offline por defecto, o DeepL).
- `cache.py` — caché de traducciones por texto normalizado.
- `tracker.py` — matching de cajas entre frames (IoU) + hash de contenido: decide qué
  recuadros reprocesar y cuáles ya están traducidos y sin cambios.
- `processor.py` — `FrameProcessor`: junta detección+OCR+traducción+tracker+caché para
  un frame.
- `server/main.py` — API FastAPI (`POST /translate`, `POST /clear-cache`, `GET /health`).

Cliente (`client/`, corre en tu PC):
- `capture.py` — hilo de lectura de la capturadora (OpenCV `VideoCapture`).
- `overlay.py` — composición del overlay (texto con acentos/kanji vía PIL) sobre el frame.
- `remote_pipeline.py` — le manda frames al servidor y guarda la última respuesta.
- `client/app.py` — ventana, loop de render, teclado.
- `client/list_devices.py` — lista índices de dispositivo de captura disponibles.

## Instalación y uso

### 1. Servidor (una sola vez, con Docker)

```bash
cp config.example.yaml config.yaml   # el servidor lee este mismo archivo
docker compose up --build -d

# Primera vez: instala los paquetes de traducción JA->EN->ES de argos-translate
# dentro del contenedor (quedan en un volumen, no hace falta repetirlo).
docker compose exec translator python -m translatens2live.setup_models
```

El primer `POST /translate` va a tardar unos segundos extra porque PaddleOCR y
manga-ocr bajan sus modelos la primera vez que se usan (después quedan cacheados en
volúmenes de Docker). Podés chequear que esté vivo con:

```bash
curl http://localhost:8000/health
```

### 2. Cliente (en tu PC, sin Docker)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-client.txt
```

1. Conectá la capturadora y confirmá qué índice de dispositivo le asigna Windows/OpenCV:
   ```bash
   python -m translatens2live.client.list_devices
   ```
2. Poné ese índice en `capture.device_index` dentro de `config.yaml`.
3. Ejecutá:
   ```bash
   python main.py
   ```

Windows es la plataforma recomendada para el cliente porque la mayoría de capturadoras
USB exponen el dispositivo como cámara UVC estándar (backend DirectShow de OpenCV). El
servidor Docker puede correr en la misma PC Windows (con Docker Desktop) sin problema,
ya que solo habla HTTP con el cliente.

Controles en la ventana del cliente:
- `q` — salir
- `o` — mostrar/ocultar overlay de traducción
- `d` — mostrar/ocultar cajas de debug (bbox verde)
- `c` — limpiar la caché de traducciones del servidor

### Fuente para el overlay

`assets/fonts/` no trae ninguna fuente versionada (ver `assets/fonts/README.md`):
bajate [Noto Sans JP](https://fonts.google.com/noto/specimen/Noto+Sans+JP) y apuntá
`overlay.font_path` en `config.yaml` al archivo. Sin esto, el overlay cae a la fuente
por defecto de PIL, que no renderiza bien kanji ni tildes.

## Desarrollo / tests

Los tests cubren la lógica pura (caché, tracker, protocolo JSON) sin necesitar los
modelos pesados instalados:

```bash
pip install -r requirements-client.txt pytest
pytest tests/ -q
```

## Estado del proyecto

MVP funcional con arquitectura cliente/servidor. Pendiente natural de siguientes
iteraciones:
- Ajustar umbrales de estabilidad/hash por juego (algunos tienen texto animado tipo
  máquina de escribir).
- Guardar la caché de traducciones periódicamente (hoy se guarda al apagar el
  servidor), por si el contenedor se cae de golpe.
- Separar cachés de traducción por juego automáticamente en vez de un único archivo.
