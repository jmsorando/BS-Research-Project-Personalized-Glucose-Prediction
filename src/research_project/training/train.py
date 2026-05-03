"""
train.py
────────
Main training script for the research_project package.
"""

import argparse
import io
import json
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from research_project import config as cfg

if __import__("sys").stdout.encoding and __import__("sys").stdout.encoding.lower() != "utf-8":
    sys_mod = __import__("sys")
    sys_mod.stdout = io.TextIOWrapper(sys_mod.stdout.buffer, encoding="utf-8", errors="replace")
    sys_mod.stderr = io.TextIOWrapper(sys_mod.stderr.buffer, encoding="utf-8", errors="replace")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data(path: Path = cfg.FEATURE_MATRIX):
    df = pd.read_csv(path)
    df = df[df["iauc_status"] == cfg.IAUC_STATUS].copy()
    df["sex"] = df["sex"].map({"Male": 0, "Female": 1})

    leakage_present = [c for c in cfg.LEAKAGE_COLS if c in cfg.ALL_FEATURES]
    assert not leakage_present, f"Leakage columns in feature list: {leakage_present}"
    missing_features, excluded_present = cfg.validate_feature_contract(df.columns)
    if missing_features:
        raise ValueError(f"Configured model features missing from dataset: {missing_features}")
    if excluded_present:
        print("[data]  excluded-by-design columns detected (not used for training):")
        print(f"        {', '.join(excluded_present)}")

    X = df[cfg.ALL_FEATURES]
    y = df[cfg.TARGET]
    groups = df["participant_id"]
    print(
        f"[data]  rows={len(df):,}  features={X.shape[1]}  "
        f"participants={groups.nunique()}  "
        f"target_mean={y.mean():.1f}  target_std={y.std():.1f}"
    )
    return X, y, groups


def cross_validate(X, y, groups, params: dict, label: str = "model") -> pd.DataFrame:
    gkf = GroupKFold(n_splits=cfg.N_FOLDS)
    records = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        model = xgb.XGBRegressor(**params)
        model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[te], y.iloc[te])], verbose=False)
        preds = model.predict(X.iloc[te])
        records.append(
            {
                "fold": fold + 1,
                "label": label,
                "MAE": mean_absolute_error(y.iloc[te], preds),
                "RMSE": np.sqrt(mean_squared_error(y.iloc[te], preds)),
                "R2": r2_score(y.iloc[te], preds),
                "n_test": len(y.iloc[te]),
                "n_test_participants": groups.iloc[te].nunique(),
                "best_iteration": model.best_iteration,
            }
        )
        print(
            f"  fold {fold+1} | MAE={records[-1]['MAE']:6.1f}  "
            f"RMSE={records[-1]['RMSE']:6.1f}  R²={records[-1]['R2']:+.3f}  "
            f"trees={model.best_iteration}"
        )
    res = pd.DataFrame(records)
    print(f"  {'─'*55}")
    print(
        f"  MEAN  | MAE={res.MAE.mean():.1f} ± {res.MAE.std():.1f}  "
        f"RMSE={res.RMSE.mean():.1f} ± {res.RMSE.std():.1f}  "
        f"R²={res.R2.mean():+.3f} ± {res.R2.std():.3f}"
    )
    return res


def tune(X, y, groups, n_trials: int = cfg.OPTUNA_N_TRIALS) -> dict:
    gkf = GroupKFold(n_splits=cfg.N_FOLDS)

    def objective(trial):
        space = cfg.OPTUNA_SEARCH_SPACE
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"][1:]),
            "max_depth": trial.suggest_int("max_depth", *space["max_depth"][1:]),
            "learning_rate": trial.suggest_float(
                "learning_rate", *space["learning_rate"][1:3], **space["learning_rate"][3]
            ),
            "subsample": trial.suggest_float("subsample", *space["subsample"][1:]),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *space["colsample_bytree"][1:]
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", *space["min_child_weight"][1:]),
            "reg_alpha": trial.suggest_float("reg_alpha", *space["reg_alpha"][1:3], **space["reg_alpha"][3]),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", *space["reg_lambda"][1:3], **space["reg_lambda"][3]
            ),
            "tree_method": "hist",
            "random_state": cfg.RANDOM_SEED,
            "early_stopping_rounds": 50,
        }
        fold_maes = []
        for tr, te in gkf.split(X, y, groups):
            m = xgb.XGBRegressor(**params)
            m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[te], y.iloc[te])], verbose=False)
            fold_maes.append(mean_absolute_error(y.iloc[te], m.predict(X.iloc[te])))
        return np.mean(fold_maes)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = {
        **study.best_params,
        "tree_method": "hist",
        "random_state": cfg.RANDOM_SEED,
        "early_stopping_rounds": 50,
    }
    print(f"\n[tune]  best MAE={study.best_value:.2f}")
    print(f"[tune]  best params: {json.dumps(study.best_params, indent=2)}")
    joblib.dump(study, cfg.MODEL_DIR / "optuna_study.pkl")
    with open(cfg.RESULTS_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"[tune]  saved → {cfg.RESULTS_DIR / 'best_params.json'}")
    return best


def train_final(X, y, params: dict) -> xgb.XGBRegressor:
    final_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    model = xgb.XGBRegressor(**final_params)
    model.fit(X, y)
    model_path = cfg.MODEL_DIR / "final_model.ubj"
    model.save_model(model_path)
    print(f"[model] saved → {model_path}")
    return model


def run_shap(model, X):
    try:
        from research_project.analysis.shap_train_exports import run_train_shap_exports

        return run_train_shap_exports(model, X, cfg.PLOT_DIR, cfg.RESULTS_DIR)
    except Exception as e:
        print(f"[shap]  SKIPPED — {e}")
        return None


def _feature_group_blocks():
    """Ordered codes match README: Dc, G, Dt, P, I."""
    return (
        ("Dc", cfg.DC_RAW + cfg.DC_RATIOS),
        ("G", cfg.G_COLS),
        ("Dt", cfg.DT_COLS),
        ("P", cfg.P_COLS),
        ("I", cfg.INTERACTION_COLS),
    )


def plot_ablation_presence_matrix(
    results: pd.DataFrame,
    out_path: Path,
    n_samples: int,
) -> None:
    """
    Matrix-style ablation table: presence dots per feature block, # features, R² (CV mean).
    Rows sorted by R² descending (best at top). Mirrors common ablation summary figures.
    """
    matrix_order = ["G", "Dc", "Dt", "P", "I"]
    col_headers = [
        "Glycemic\n(G)",
        "Diet comp.\n(Dc)",
        "Diet temporal\n(Dt)",
        "Personal\n(P)",
        "Interactions\n(I)",
    ]

    df = results.sort_values("R2_mean", ascending=False).reset_index(drop=True)
    n_rows = len(df)
    text_color = "#1a1a1a"

    fig_w, fig_h = 10.0, max(6.0, 0.32 * n_rows + 2.4)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.07, 0.06, 0.68, 0.78])

    dx = 1.0
    x_dots = np.arange(len(matrix_order), dtype=float) * dx
    present_fc = "#41ab5d"
    present_ec = "#238b45"
    absent_fc = "#e8e8e8"
    absent_ec = "#bdbdbd"

    for i, (_, row) in enumerate(df.iterrows()):
        y = float(n_rows - 1 - i)
        parts = [p.strip() for p in str(row["label"]).split(" + ")]
        active = set(parts)

        ax.hlines(y, x_dots[0], x_dots[-1], colors="#f3f3f3", linewidths=5, zorder=0)

        for xi, code in zip(x_dots, matrix_order):
            if code in active:
                ax.scatter(
                    xi,
                    y,
                    s=130,
                    c=present_fc,
                    edgecolors=present_ec,
                    linewidths=0.9,
                    zorder=2,
                )
            else:
                ax.scatter(
                    xi,
                    y,
                    s=130,
                    c=absent_fc,
                    edgecolors=absent_ec,
                    linewidths=0.6,
                    zorder=1,
                )

        ordered = [c for c in matrix_order if c in active]
        latex = "$" + "+".join(ordered) + "$"
        ax.text(x_dots[0] - 0.65, y, latex, ha="right", va="center", fontsize=9)

    x_nf = float(x_dots[-1] + 1.2)
    x_r2 = float(x_dots[-1] + 2.55)

    ax.text(x_nf, n_rows + 0.42, "# features", ha="center", fontsize=8, fontweight="bold")
    ax.text(x_r2, n_rows + 0.42, r"$R^2$ (mean CV)", ha="center", fontsize=8, fontweight="bold")

    for i, (_, row) in enumerate(df.iterrows()):
        y = float(n_rows - 1 - i)
        vnf = int(row["n_features"])
        ax.text(
            x_nf,
            y,
            str(vnf),
            ha="center",
            va="center",
            fontsize=9,
            color=text_color,
            fontweight="bold",
        )
        vr2 = float(row["R2_mean"])
        ax.text(
            x_r2,
            y,
            f"{vr2:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            color=text_color,
            fontweight="bold",
        )

    ax.set_xlim(x_dots[0] - 2.4, x_r2 + 0.45)
    ax.set_ylim(-0.75, n_rows + 0.55)
    ax.set_xticks(x_dots)
    ax.set_xticklabels(col_headers, fontsize=8)
    ax.tick_params(left=False, labelleft=False, bottom=True)
    for s in ax.spines.values():
        s.set_visible(False)

    fig.text(
        0.07,
        0.935,
        f"Same cohort for all rows: {n_samples:,} meals (GroupKFold CV by participant)",
        fontsize=9,
        style="italic",
        ha="left",
        va="top",
    )

    leg = fig.add_axes([0.80, 0.38, 0.18, 0.12])
    leg.set_axis_off()
    leg.scatter([0.06], [0.72], s=70, c=present_fc, edgecolors=present_ec, transform=leg.transAxes)
    leg.text(0.14, 0.72, "Feature set present", transform=leg.transAxes, va="center", fontsize=8)
    leg.scatter([0.06], [0.28], s=70, c=absent_fc, edgecolors=absent_ec, transform=leg.transAxes)
    leg.text(0.14, 0.28, "Feature set absent", transform=leg.transAxes, va="center", fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ablation(X_full, y, groups, params: dict) -> pd.DataFrame:
    blocks = _feature_group_blocks()
    group_order = [code for code, _ in blocks]
    code_to_feats = dict(blocks)

    feature_sets: list[tuple[str, list]] = []
    for r in range(1, len(group_order) + 1):
        for combo in combinations(group_order, r):
            feats: list = []
            for code in combo:
                feats.extend(code_to_feats[code])
            label = " + ".join(combo)
            feature_sets.append((label, feats))

    gkf = GroupKFold(n_splits=cfg.N_FOLDS)
    rows = []
    for label, feats in feature_sets:
        Xs = X_full[feats]
        r2s, maes, rmses = [], [], []
        for tr, te in gkf.split(Xs, y, groups):
            m = xgb.XGBRegressor(**params)
            m.fit(Xs.iloc[tr], y.iloc[tr], eval_set=[(Xs.iloc[te], y.iloc[te])], verbose=False)
            preds = m.predict(Xs.iloc[te])
            y_te = y.iloc[te]
            r2s.append(r2_score(y_te, preds))
            maes.append(mean_absolute_error(y_te, preds))
            rmses.append(np.sqrt(mean_squared_error(y_te, preds)))
        row = {
            "label": label,
            "n_features": len(feats),
            "R2_mean": np.mean(r2s),
            "R2_std": np.std(r2s),
            "MAE_mean": np.mean(maes),
            "MAE_std": np.std(maes),
            "RMSE_mean": np.mean(rmses),
            "RMSE_std": np.std(rmses),
        }
        rows.append(row)
        print(
            f"  {label:22s} | n={len(feats):3d} | "
            f"R²={row['R2_mean']:+.3f} ± {row['R2_std']:.3f} | "
            f"MAE={row['MAE_mean']:.1f} ± {row['MAE_std']:.1f} | "
            f"RMSE={row['RMSE_mean']:.1f}"
        )
    results = pd.DataFrame(rows)
    results = results.sort_values("R2_mean", ascending=True).reset_index(drop=True)

    fig_h = max(5.0, 0.22 * len(results) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    colours = ["#4472C4" if r > 0 else "#C0504D" for r in results.R2_mean]
    y_pos = np.arange(len(results))
    ax.barh(y_pos, results.R2_mean, xerr=results.R2_std, color=colours, capsize=2)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results.label, fontsize=8)
    ax.set_xlabel("R² (10-fold GroupKFold CV)")
    ax.set_title("Ablation — all feature-group combinations (Dc, G, Dt, P, I)")
    xmax = max(results.R2_mean.max() + results.R2_std.max(), 0.05)
    for i, row in results.iterrows():
        ax.text(min(row.R2_mean + row.R2_std, xmax) + 0.005, i, f"{row.R2_mean:+.3f}", va="center", fontsize=7)
    plt.tight_layout()
    p = cfg.PLOT_DIR / "ablation.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ablation] saved → {p}")

    ablation_csv = cfg.RESULTS_DIR / "ablation_by_feature_group.csv"
    sorted_down = results.sort_values("R2_mean", ascending=False).reset_index(drop=True)
    sorted_down.to_csv(ablation_csv, index=False)
    print(f"[ablation] table → {ablation_csv}")

    p_matrix = cfg.PLOT_DIR / "ablation_matrix.png"
    plot_ablation_presence_matrix(sorted_down, p_matrix, len(y))
    print(f"[ablation] matrix figure → {p_matrix}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="iAUC XGBoost training pipeline")
    p.add_argument("--tune", action="store_true", help="Run Optuna tuning")
    p.add_argument("--shap", action="store_true", help="Run SHAP analysis")
    p.add_argument("--ablation", action="store_true", help="Run ablation study")
    p.add_argument("--all", action="store_true", help="Run everything")
    p.add_argument("--data", type=str, default=None, help="Override data path")
    return p.parse_args()


def main():
    args = parse_args()
    if args.all:
        args.tune = args.shap = args.ablation = True

    data_path = Path(args.data) if args.data else cfg.FEATURE_MATRIX
    X, y, groups = load_data(data_path)
    all_rows = []

    print("\n" + "═" * 60)
    print("BASELINE CROSS-VALIDATION")
    print("═" * 60)
    baseline_results = cross_validate(X, y, groups, cfg.BASELINE_PARAMS, label="baseline")
    all_rows.append(
        {
            "result_type": "cv_summary",
            "parameter_source": "baseline",
            "label": "baseline",
            "n_features": X.shape[1],
            "R2_mean": baseline_results.R2.mean(),
            "R2_std": baseline_results.R2.std(),
            "MAE_mean": baseline_results.MAE.mean(),
            "MAE_std": baseline_results.MAE.std(),
            "RMSE_mean": baseline_results.RMSE.mean(),
            "RMSE_std": baseline_results.RMSE.std(),
        }
    )

    active_params = cfg.BASELINE_PARAMS
    if args.tune:
        print("\n" + "═" * 60)
        print("HYPERPARAMETER TUNING")
        print("═" * 60)
        best_params = tune(X, y, groups)
        print("\nTUNED MODEL CV")
        print("─" * 60)
        tuned_results = cross_validate(X, y, groups, best_params, label="tuned")
        all_rows.append(
            {
                "result_type": "cv_summary",
                "parameter_source": "tuned",
                "label": "tuned",
                "n_features": X.shape[1],
                "R2_mean": tuned_results.R2.mean(),
                "R2_std": tuned_results.R2.std(),
                "MAE_mean": tuned_results.MAE.mean(),
                "MAE_std": tuned_results.MAE.std(),
                "RMSE_mean": tuned_results.RMSE.mean(),
                "RMSE_std": tuned_results.RMSE.std(),
            }
        )
        active_params = best_params

    print("\n" + "═" * 60)
    print("TRAINING FINAL MODEL (all data)")
    print("═" * 60)
    final_model = train_final(X, y, active_params)

    if args.shap:
        print("\n" + "═" * 60)
        print("SHAP ANALYSIS")
        print("═" * 60)
        run_shap(final_model, X)

    if args.ablation:
        print("\n" + "═" * 60)
        print("ABLATION STUDY")
        print("═" * 60)
        ablation_results = ablation(X, y, groups, active_params)
        param_source = "tuned" if args.tune else "baseline"
        for _, row in ablation_results.iterrows():
            row_dict = row.to_dict()
            all_rows.append(
                {
                    "result_type": "ablation",
                    "parameter_source": param_source,
                    **row_dict,
                }
            )

    results_df = pd.DataFrame(all_rows)
    preferred_order = [
        "result_type",
        "parameter_source",
        "label",
        "n_features",
        "R2_mean",
        "R2_std",
        "MAE_mean",
        "MAE_std",
        "RMSE_mean",
        "RMSE_std",
    ]
    ordered_cols = [c for c in preferred_order if c in results_df.columns] + [
        c for c in results_df.columns if c not in preferred_order
    ]
    results_df = results_df[ordered_cols]
    results_df.to_csv(cfg.RESULTS_DIR / "results.csv", index=False)
    print("\n✓ Pipeline complete. Outputs in:", cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()

