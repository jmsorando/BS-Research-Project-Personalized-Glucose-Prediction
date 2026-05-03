"""Lightweight SHAP exports used by `train.py --shap` (summary + top-4 dependence)."""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

_NUMBA_MOCKS = [
    "numba",
    "numba.core",
    "numba.core.decorators",
    "numba.stencils",
    "numba.stencils.stencil",
    "numba.core.ir_utils",
    "numba.core.extending",
    "numba.core.pythonapi",
    "numba.typed",
]


def run_train_shap_exports(
    model,
    X: pd.DataFrame,
    plot_dir: Path,
    results_dir: Path,
) -> np.ndarray:
    """
    TreeExplainer on full data; writes `shap_values.csv`, `shap_summary.png`,
    and `shap_dependence_top4.png` under plot_dir (flat — not plots/shap/).
    """
    for mod in _NUMBA_MOCKS:
        if mod not in sys.modules:
            sys.modules[mod] = unittest.mock.MagicMock()

    import shap

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("[shap]  computing SHAP values ...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    pd.DataFrame(shap_vals, columns=X.columns).to_csv(results_dir / "shap_values.csv", index=False)

    shap.summary_plot(shap_vals, X, max_display=20, show=False)
    plt.gca().set_xlabel("SHAP value (mmol·min/L)")
    plt.tight_layout()
    p_sum = plot_dir / "shap_summary.png"
    plt.savefig(p_sum, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap]  saved -> {p_sum}")

    mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=X.columns)
    top4 = mean_abs.nlargest(4).index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, feat in zip(axes.flat, top4):
        shap.dependence_plot(feat, shap_vals, X, ax=ax, show=False)
        ax.set_ylabel("SHAP value (mmol·min/L)")
    plt.tight_layout()
    p_dep = plot_dir / "shap_dependence_top4.png"
    plt.savefig(p_dep, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap]  saved -> {p_dep}")
    return shap_vals
