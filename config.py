"""Compatibility shim: import config from `src/rp_glucose`."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rp_glucose.config import *  # noqa: F401,F403
