"""Compatibility entrypoint: runs the full SHAP report in `research_project.analysis.shap_analysis`."""

import bootstrap_path  # noqa: E402

from research_project.analysis.shap_analysis import main


if __name__ == "__main__":
    main()
