"""Compatibility entrypoint: delegates to src/rp_glucose.analysis.shap_analysis.py."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rp_glucose.analysis.shap_analysis import main


if __name__ == "__main__":
    main()
