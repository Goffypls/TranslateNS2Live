"""Entrypoint de conveniencia: `python main.py [config.yaml]`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from translatens2live.app import main  # noqa: E402

if __name__ == "__main__":
    main()
