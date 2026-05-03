"""
plots.py
--------
Publication figures and out-of-fold diagnostics.
"""

import argparse
import json
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from research_project import config as cfg

UNIT = "mmol·min/L"


def plot_iauc_distribution() -> None:
    df = cast(pd.DataFrame, pd.read_csv(cfg.FEATURE_MATRIX))
    df = df[df["iauc_status"] == cfg.IAUC_STATUS].copy()
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


def compute_oof_predictions() -> pd.DataFrame:
    df = cast(pd.DataFrame, pd.read_csv(cfg.FEATURE_MATRIX))
    df = df[df["iauc_status"] == cfg.IAUC_STATUS].reset_index(drop=True)
    df["sex"] = cast(pd.Series, df["sex"]).map({"Male": 0, "Female": 1})

    X = cast(pd.DataFrame, df[cfg.ALL_FEATURES])
    y = cast(pd.Series, df[cfg.TARGET])
    groups = cast(pd.Series, df["participant_id"]).to_numpy()

    best_params_path = cfg.RESULTS_DIR / "best_params.json"
    if best_params_path.exists():
        with open(best_params_path, encoding="utf-8") as f:
            params = json.load(f)
        print(f"[params] loaded tuned params from {best_params_path}")
    else:
        params = dict(cfg.BASELINE_PARAMS)
        print("[params] using BASELINE_PARAMS")

    gkf = GroupKFold(n_splits=cfg.N_FOLDS)
    oof_pred = np.full(len(df), np.nan)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[te], y.iloc[te])], verbose=False)
        oof_pred[te] = m.predict(X.iloc[te])
        print(f"  fold {fold:2d} | n_test={len(te):4d} | MAE={mean_absolute_error(y.iloc[te], oof_pred[te]):.2f}")

    oof = pd.DataFrame(
        {
            "participant_id": groups,
            "actual_iauc": y.values,
            "predicted_iauc": oof_pred,
        }
    )
    out_csv = cfg.RESULTS_DIR / "oof_predictions.csv"
    oof.to_csv(out_csv, index=False)
    print(f"[oof] saved -> {out_csv}")
    return oof


def plot_oof_scatter(oof: pd.DataFrame, r2: float, mae: float) -> None:
    point_colour = "#2166AC"
    x = np.asarray(oof["actual_iauc"].to_numpy(), dtype=float)
    yhat = np.asarray(oof["predicted_iauc"].to_numpy(), dtype=float)
    lo = float(min(x.min(), yhat.min()))
    hi = float(max(x.max(), yhat.max()))
    pad = 0.05 * (hi - lo)
    lims = (lo - pad, hi + pad)

    slope_v, intercept_v, _, _, _ = stats.linregress(x, yhat)
    slope = float(slope_v)
    intercept = float(intercept_v)
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
        f"$R^2$ = {r2:.3f}\nMAE = {mae:.1f} {UNIT}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", lw=0.8, alpha=0.9),
    )
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    plt.tight_layout()
    out = cfg.PLOT_DIR / "oof_scatter.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved -> {out}")


def run_oof_scatter() -> None:
    oof = compute_oof_predictions()
    r2 = r2_score(oof["actual_iauc"], oof["predicted_iauc"])
    mae = mean_absolute_error(oof["actual_iauc"], oof["predicted_iauc"])
    print(f"[oof] R²={r2:.3f}  MAE={mae:.2f}")
    plot_oof_scatter(oof, r2, mae)


def main() -> None:
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

