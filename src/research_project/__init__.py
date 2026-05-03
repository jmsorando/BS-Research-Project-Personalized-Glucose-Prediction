"""
research_project package scaffold (Phase 1).
"""

import sys

# Ensure UTF-8 stdout/stderr on Windows (cp1252 chokes on scientific chars).
# Done once here so every submodule inherits it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
