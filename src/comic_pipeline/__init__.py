"""NaNoBananacomic local pipeline package."""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["__version__"]

__version__ = "0.1.0"


def _add_local_dependency_dirs() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    extra_dirs = os.environ.get("COMIC_PIPELINE_EXTRA_VENDOR_DIRS", "")
    candidates: list[Path] = []
    if extra_dirs:
        candidates.extend(Path(raw) for raw in extra_dirs.split(os.pathsep) if raw.strip())
    candidates.extend([repo_root / ".vendor", repo_root / ".bootstrap"])

    for candidate in candidates:
        if candidate.exists():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)


_add_local_dependency_dirs()
