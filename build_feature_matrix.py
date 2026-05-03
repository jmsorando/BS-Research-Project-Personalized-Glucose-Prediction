"""Compatibility entrypoint: delegates to `src/research_project.data_pipeline.feature_matrix.py`."""

import bootstrap_path  # noqa: E402

from research_project.data_pipeline.feature_matrix import main


if __name__ == "__main__":
    main()
