"""Shared SHAP helpers: numba mocking, TreeExplainer, optional CSV cache."""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path

import numpy as np
import pandas as pd

NUMBA_SHAP_MOCK_MODULES: tuple[str, ...] = (
    "numba",
    "numba.core",
    "numba.core.decorators",
    "numba.stencils",
    "numba.stencils.stencil",
    "numba.core.ir_utils",
    "numba.core.extending",
    "numba.core.pythonapi",
    "numba.typed",
)


def mock_numba_for_shap() -> None:
    for mod in NUMBA_SHAP_MOCK_MODULES:
        if mod not in sys.modules:
            sys.modules[mod] = unittest.mock.MagicMock()


def _expected_value_scalar(explainer) -> float:
    ev = explainer.expected_value
    if hasattr(ev, "__len__") and not isinstance(ev, str):
        ev_seq = ev
        ev = float(ev_seq[0]) if len(ev_seq) == 1 else float(np.mean(ev_seq))
    return float(ev)


def shap_matrix_and_expected(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
    shap_csv: Path,
    *,
    prefer_disk: bool = True,
) -> tuple[np.ndarray, float, bool]:
    """
    One TreeExplainer per call. If ``prefer_disk`` and ``shap_csv`` is compatible
    with ``X``, load the matrix from disk; otherwise compute, save, return.

    Returns ``(shap_values, expected_value, loaded_from_disk)``.
    """
    mock_numba_for_shap()
    import shap

    explainer = shap.TreeExplainer(model)
    ev = _expected_value_scalar(explainer)

    if prefer_disk and shap_csv.exists():
        existing = pd.read_csv(shap_csv)
        if (
            list(existing.columns) == list(feature_names)
            and len(existing) == len(X)
            and existing.shape[1] == len(feature_names)
        ):
            return existing.to_numpy(dtype=float, copy=False), ev, True

    shap_vals = explainer.shap_values(X)
    pd.DataFrame(shap_vals, columns=feature_names).to_csv(shap_csv, index=False)
    return shap_vals, ev, False
