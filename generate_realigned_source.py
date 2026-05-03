"""Compatibility entrypoint: delegates to `src/research_project.data_pipeline.build_realigned_source.py`."""

import bootstrap_path  # noqa: E402

from research_project.data_pipeline.build_realigned_source import main


if __name__ == "__main__":
    main()
