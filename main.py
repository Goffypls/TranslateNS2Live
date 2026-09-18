"""Entrypoint de conveniencia del cliente: `python main.py [config.yaml]`.

Esto es el cliente liviano (captura + overlay). El motor de traducción
corre aparte, en Docker (ver README / docker-compose.yml)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from translatens2live.client.app import main  # noqa: E402

if __name__ == "__main__":
    main()
