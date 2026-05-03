"""Prepend repo `src/` to `sys.path` so root scripts can import `research_project`."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
_src_str = str(_SRC)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)
