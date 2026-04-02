"""NaNoBananacomic local pipeline package."""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["__version__"]

__version__ = "0.1.0"


def _add_local_dependency_dirs() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for name in (".vendor", ".bootstrap"):
        candidate = repo_root / name
        if candidate.exists():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)


_add_local_dependency_dirs()
