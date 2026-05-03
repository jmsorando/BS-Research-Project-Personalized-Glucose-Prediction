"""Compatibility entrypoint: delegates to `src/research_project/training/train.py`."""

import bootstrap_path  # noqa: E402

from research_project.training.train import main


if __name__ == "__main__":
    main()
