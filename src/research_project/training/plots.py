"""
Publication figures and out-of-fold diagnostics.

Run: ``rp-plots`` or ``python -m research_project.training.plots`` (after ``pip install -e .``).
"""

import argparse
import json
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from research_project import config as cfg
from research_project.training.train import (
    FULL_FEATURE_GROUP_ABLATION_LABEL,
    LAST_FIT_PARAMS_JSON,
    groupkfold_xgb_oof_and_fold_metrics,
    load_data,
    load_training_table,
)

UNIT = "mmol·min/L"


def plot_iauc_distribution() -> None:
    df = load_training_table()
    y = cast(pd.Series, df[cfg.TARGET])

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(y, bins=40, color="#4472C4", edgecolor="white", linewidth=0.3)
    ax1.axvline(y.mean(), color="red", linestyle="--", linewidth=1.2, label="Mean")
    ax1.axvline(y.median(), color="orange", linestyle="--", linewidth=1.2, label="Median")
    ax1.set_xlabel(f"iAUC ({UNIT})")
    ax1.set_ylabel("Number of meal events")
    ax1.set_title("(a) iAUC distribution")
    ax1.legend(frameon=False)

    participants = sorted(cast(pd.Series, df["participant_id"]).unique())[:15]
    groups = [df.loc[df["participant_id"] == p, cfg.TARGET].values for p in participants]
    ax2.boxplot(groups, labels=participants, showfliers=True)
    ax2.set_xlabel("Participant ID")
    ax2.set_ylabel(f"iAUC ({UNIT})")
    ax2.set_title("(b) iAUC per participant (sample)")
    ax2.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    for ext in ("png", "svg"):
        p = cfg.PLOT_DIR / f"pub_figure1_iAUC_distribution.{ext}"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close()


def _params_for_oof_scatter() -> dict:
    """Params from the last ``rp-train`` run (matches ablation); else tuned snapshot; else baseline."""
    merged = dict(cfg.BASELINE_PARAMS)
    if LAST_FIT_PARAMS_JSON.exists():
        with open(LAST_FIT_PARAMS_JSON, encoding="utf-8") as f:
            merged.update(json.load(f))
        print(f"[params] using last training run → {LAST_FIT_PARAMS_JSON}")
        return merged
    best_params_path = cfg.RESULTS_DIR / "best_params.json"
    if best_params_path.exists():
        with open(best_params_path, encoding="utf-8") as f:
            merged.update(json.load(f))
        print(
            f"[params] no {LAST_FIT_PARAMS_JSON.name}; loaded {best_params_path.name} "
            "(run rp-train so OOF matches ablation from the same run)"
        )
        return merged
    print("[params] using BASELINE_PARAMS only")
    return merged


def compute_oof_predictions() -> tuple[pd.DataFrame, float, float]:
    """Fit GroupKFold OOF preds; return (oof frame, mean per-fold R², mean per-fold MAE).

    Uses the same CV loop as ``train.ablation`` (``groupkfold_xgb_oof_and_fold_metrics``)
    and the same hyperparameters as the last ``rp-train`` (``last_fit_params.json``) so
    mean CV R²/MAE match the matrix row ``Dc + G + Dt + P + I`` from that run.
    """
    X, y, groups = load_data()
    groups_arr = groups.to_numpy()
    params = _params_for_oof_scatter()

    oof_pred, r2_folds, mae_folds, _ = groupkfold_xgb_oof_and_fold_metrics(X, y, groups, params)
    for fold, (mae, r2) in enumerate(zip(mae_folds, r2_folds, strict=True), start=1):
        print(f"  fold {fold:2d} | MAE={mae:.2f}  R²={r2:+.3f}")

    oof = pd.DataFrame(
        {
            "participant_id": groups_arr,
            "actual_iauc": y.values,
            "predicted_iauc": oof_pred,
        }
    )
    out_csv = cfg.RESULTS_DIR / "oof_predictions.csv"
    oof.to_csv(out_csv, index=False)
    print(f"[oof] saved -> {out_csv}")
    r2_mean_cv = float(np.mean(r2_folds))
    mae_mean_cv = float(np.mean(mae_folds))
    return oof, r2_mean_cv, mae_mean_cv


def plot_oof_scatter(oof: pd.DataFrame, r2: float, mae: float) -> None:
    point_colour = "#2166AC"
    x = np.asarray(oof["actual_iauc"].to_numpy(), dtype=float)
    yhat = np.asarray(oof["predicted_iauc"].to_numpy(), dtype=float)
    lo = float(min(x.min(), yhat.min()))
    hi = float(max(x.max(), yhat.max()))
    pad = 0.05 * (hi - lo)
    lims = (lo - pad, hi + pad)

    # Stubs widen linregress components to object; cast before float().
    lr = stats.linregress(x, yhat)
    slope = float(cast(Any, lr[0]))
    intercept = float(cast(Any, lr[1]))
    xx = np.linspace(lims[0], lims[1], 200)
    yy = slope * xx + intercept

    n = len(x)
    resid = yhat - (slope * x + intercept)
    sigma = np.sqrt(np.sum(resid**2) / (n - 2))
    x_mean = x.mean()
    ssx = np.sum((x - x_mean) ** 2)
    se_fit = sigma * np.sqrt(1.0 / n + (xx - x_mean) ** 2 / ssx)
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ci = t_crit * se_fit

    _, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(lims, lims, color="grey", lw=1.2, ls="--", label="y = x", zorder=1)
    ax.scatter(x, yhat, c=point_colour, s=18, alpha=0.5, edgecolors="none", zorder=2)
    ax.fill_between(xx, yy - ci, yy + ci, color="#D6604D", alpha=0.25, zorder=3)
    ax.plot(xx, yy, color="#B2182B", lw=2.0, label="OLS fit", zorder=4)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"Observed iAUC ({UNIT})", fontsize=12)
    ax.set_ylabel(f"Predicted iAUC ({UNIT})", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4)
    ax.text(
        0.04,
        0.96,
        f"$R^2$ (mean CV) = {r2:.3f}\nMAE (mean CV) = {mae:.1f} {UNIT}\n"
        f"({FULL_FEATURE_GROUP_ABLATION_LABEL})",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", lw=0.8, alpha=0.9),
    )
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    plt.tight_layout()
    out = cfg.PLOT_DIR / "oof_scatter.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved -> {out}")


def run_oof_scatter() -> None:
    oof, r2_mean_cv, mae_mean_cv = compute_oof_predictions()
    print(f"[oof] R² (mean CV)={r2_mean_cv:.3f}  MAE (mean CV)={mae_mean_cv:.2f}")
    plot_oof_scatter(oof, r2_mean_cv, mae_mean_cv)


def main() -> None:
    cfg.ensure_output_dirs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "which",
        choices=["iauc-dist", "oof-scatter", "all"],
        nargs="?",
        default="all",
        help="Which figure to generate (default: all)",
    )
    args = parser.parse_args()
    if args.which in ("iauc-dist", "all"):
        plot_iauc_distribution()
    if args.which in ("oof-scatter", "all"):
        run_oof_scatter()


if __name__ == "__main__":
    main()

