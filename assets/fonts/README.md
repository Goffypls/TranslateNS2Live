Poné acá una fuente TTF/OTF que soporte japonés + acentos en español, por ejemplo
[Noto Sans JP](https://fonts.google.com/noto/specimen/Noto+Sans+JP), y apuntá
`overlay.font_path` en tu `config.yaml` al archivo descargado
(ej: `assets/fonts/NotoSansJP-Regular.otf`).

No se versiona el binario de la fuente en el repo (ver `.gitignore`); si no hay
ninguna fuente en esta carpeta, la app cae a la fuente por defecto de PIL, que
no renderiza bien kanji ni tildes.
