"""Lightweight SHAP exports used by ``rp-train --shap`` (summary + top-4 dependence)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from research_project import config as cfg
from research_project.analysis.shap_support import shap_matrix_and_expected


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
    import shap

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    shap_csv = results_dir / "shap_values.csv"
    model_path = cfg.MODEL_DIR / "final_model.ubj"
    prefer_disk = True
    if shap_csv.exists() and model_path.exists():
        if model_path.stat().st_mtime > shap_csv.stat().st_mtime:
            prefer_disk = False
            print("[shap]  model newer than SHAP cache — recomputing SHAP values ...")
    else:
        prefer_disk = False
        print("[shap]  computing SHAP values ...")
    if prefer_disk:
        print("[shap]  loading SHAP values from cache (if compatible) ...")
    shap_vals, _, _ = shap_matrix_and_expected(
        model,
        X,
        list(X.columns),
        shap_csv,
        prefer_disk=prefer_disk,
    )

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
