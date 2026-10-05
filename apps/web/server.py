"""Recommended web entry point.

The legacy ``frontend/server.py`` remains importable for existing scripts and
tests. This wrapper gives the reorganized project a stable launch path.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from frontend.server import main


if __name__ == "__main__":
    main()
