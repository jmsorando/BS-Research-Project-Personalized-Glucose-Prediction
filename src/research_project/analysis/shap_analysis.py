"""
Comprehensive SHAP analysis for the iAUC XGBoost model.
Produces summary tables plus figures 01–04 (CV fold stability is CSV-only, between figs 03 and 04).

After `pip install -e .`:
    rp-shap-report
    # or: python -m research_project.analysis.shap_analysis
"""

import sys
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold

from research_project import config as cfg
from research_project.analysis.shap_support import shap_matrix_and_expected
from research_project.training.train import load_data

# ── Colour palette (ColorBrewer, colorblind-safe, print-safe) ────────────
COLOUR = {
    "G":  "#2166AC",   # blue — glycaemic context
    "Dc": "#D6604D",   # red-orange — diet composition
    "Dt": "#4DAC26",   # green — diet temporal
    "P":  "#7B2D8B",   # purple — participant covariates
    "I":  "#DDCC77",   # Tol muted gold — interactions
}

# ── Feature units (for axis labels) ─────────────────────────────────────────
_G_PER_MEAL = "g"
_MMOL_L = "mmol/L"

FEATURE_UNITS = {
    # DC_RAW — macronutrients & sugars (g per meal)
    **{f: _G_PER_MEAL for f in [
        "CHO", "STAR", "TOTSUG", "FREE_SUGAR", "ADDED_SUGAR",
        "GLUC", "FRUCT", "SUCR", "MALT", "LACT", "GALACT", "OLIGO",
        "PROT", "FAT", "ALCO", "WATER",
        "AOACFIB", "ENGFIB",
        "SATFAC", "MONOFACc", "POLYFACc", "TOTn3PFAC", "TOTn6PFAC", "FACTRANS",
        "totalVeg", "totalFruit",
    ]},
    "KCALS": "kcal",
    "MG": "mg", "ZN": "mg", "MN": "mg", "FE": "mg", "CAFF": "mg",
    "SE": "µg", "VITD": "µg",
    # DC_RATIOS — dimensionless
    **{f: "g/g" for f in [
        "starch_fraction", "sugar_fraction", "free_sugar_fraction",
        "protein_cho_ratio", "fat_sugar_ratio", "protein_sugar_ratio",
        "n6_n3_ratio",
    ]},
    "rapid_glucose_equiv": _G_PER_MEAL,
    "intrinsic_sugar": _G_PER_MEAL,
    "glycaemic_brake": "weighted g/g",
    # G_COLS — glycaemic context
    "baseline_glucose_mmol": _MMOL_L,
    "past_1h_glucose_mean": _MMOL_L,
    "past_1h_glucose_sd": _MMOL_L,
    "mean_glucose_24h": _MMOL_L,
    "sd_glucose_24h": _MMOL_L,
    "mage_24h": _MMOL_L,
    "conga1_24h": _MMOL_L,
    "conga2_24h": _MMOL_L,
    "modd_24h": _MMOL_L,
    "glucose_at_t_minus_15": _MMOL_L,
    "glucose_at_t_minus_30": _MMOL_L,
    "past_4h_glucose_trend": "mmol/L/h",
    # DT_COLS — diet temporal
    "past_3h_cho": _G_PER_MEAL, "past_3h_sugar": _G_PER_MEAL,
    "past_3h_fat": _G_PER_MEAL, "past_3h_prot": _G_PER_MEAL,
    "time_since_last_meal_min": "min",
    "hour_of_day": "h",
    # P_COLS
    "sex": "0=M / 1=F",
    # INTERACTION_COLS
    "cho_x_baseline_glucose": "g·mmol/L",
    "cho_x_mage": "g·mmol/L",
    "cho_x_time_since_last_meal": "g·min",
    "cho_x_fibre": "g²",
    "cho_x_fat": "g²",
    "cho_x_protein": "g²",
    "cho_x_hour_of_day": "g·h",
    "glucose_trend_x_hour_of_day": "mmol/L",
    "mage_x_baseline_glucose": "(mmol/L)²",
}




def get_feature_group(feat):
    if feat in cfg.DC_RAW or feat in cfg.DC_RATIOS:
        return "Dc"
    elif feat in cfg.G_COLS:
        return "G"
    elif feat in cfg.DT_COLS:
        return "Dt"
    elif feat in cfg.P_COLS:
        return "P"
    elif feat in cfg.INTERACTION_COLS:
        return "I"
    return "?"


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: rp-shap-report  |  python -m research_project.analysis.shap_analysis")
        print("  Full SHAP report (figures 01–04, CSVs). Requires feature matrix and trained model.")
        return

    cfg.ensure_output_dirs()
    shap_plot_dir = cfg.PLOT_DIR / "shap"
    shap_plot_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════════

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    X, y, groups = load_data()
    participant_ids = groups.to_numpy()
    print(f"  Rows: {len(X):,}  Features: {X.shape[1]}  "
          f"Participants: {groups.nunique()}")


    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL & SHAP VALUES
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("LOADING / TRAINING MODEL")
    print("=" * 60)

    model_path = cfg.MODEL_DIR / "final_model.ubj"
    shap_csv   = cfg.RESULTS_DIR / "shap_values.csv"
    params_json = cfg.RESULTS_DIR / "best_params.json"

    # Try to load best params; fall back to baseline
    if params_json.exists():
        with open(params_json) as f:
            best_params = json.load(f)
        print(f"  Loaded best_params from {params_json}")
    else:
        best_params = dict(cfg.BASELINE_PARAMS)
        print("  No best_params.json found — using BASELINE_PARAMS")

    train_params = {k: v for k, v in best_params.items() if k != "early_stopping_rounds"}

    # Load or train model
    if model_path.exists():
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        print(f"  Loaded model from {model_path}")
    else:
        print("  No saved model found — training final model on all data ...")
        model = xgb.XGBRegressor(**train_params)
        model.fit(X, y)
        model.save_model(str(model_path))
        print(f"  Saved model to {model_path}")

    prefer_disk = True
    if shap_csv.exists() and model_path.exists():
        if model_path.stat().st_mtime > shap_csv.stat().st_mtime:
            prefer_disk = False
            print("  Model newer than SHAP cache — recomputing SHAP values")

    print("  Resolving SHAP values (compatible cache → load; else TreeExplainer) ...")
    shap_vals, expected_value, from_disk = shap_matrix_and_expected(
        model, X, list(cfg.ALL_FEATURES), shap_csv, prefer_disk=prefer_disk
    )
    print(f"  {'Loaded' if from_disk else 'Saved'} SHAP values → {shap_csv.name}")

    print(f"  SHAP matrix shape: {shap_vals.shape}")
    print(f"  Expected value (baseline): {expected_value:.2f}")


    feature_groups = [get_feature_group(f) for f in cfg.ALL_FEATURES]

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    feat_importance = pd.Series(mean_abs_shap, index=cfg.ALL_FEATURES)
    dc_all = cfg.DC_RAW + cfg.DC_RATIOS

    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYSIS 1 — Beeswarm Summary (Top 20)
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("ANALYSIS 1 — Beeswarm Summary (Top 20)")
    print("=" * 60)

    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_vals, X, max_display=20, show=False)
    plt.gca().set_xlabel("SHAP value (mmol·min/L)")
    plt.title("SHAP Global Feature Importance", fontsize=12)
    plt.tight_layout()
    p = shap_plot_dir / "01_beeswarm_top20.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")
    print("  Caption: Beeswarm plot showing top 20 features ranked by mean |SHAP|.")


    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYSIS 2 — Dependence Plots for Top 4 Features → 02_*.png
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("ANALYSIS 2 — Dependence Plots (Top 4)")
    print("=" * 60)

    top4_dep = feat_importance.nlargest(4).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, (ax, feat) in enumerate(zip(axes.flat, top4_dep)):
        shap.dependence_plot(feat, shap_vals, X, interaction_index="auto",
                             ax=ax, show=False)
        unit = FEATURE_UNITS.get(feat, "")
        ax.set_xlabel(f"{feat} ({unit})" if unit else feat)
        ax.set_ylabel("SHAP value (mmol·min/L)")
        ax.set_title(f"#{i+1} {feat}", fontsize=10)
    for cb_ax in fig.axes:
        if cb_ax not in axes.flat:
            label = cb_ax.get_ylabel()
            cb_unit = FEATURE_UNITS.get(label, "")
            if cb_unit:
                cb_ax.set_ylabel(f"{label} ({cb_unit})")
    plt.suptitle("SHAP Dependence Plots — Top 4 Features", fontsize=13, y=1.01)
    plt.tight_layout()
    p = shap_plot_dir / "02_dependence_top4.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")
    print(f"  Top 4: {top4_dep}")


    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYSIS 3 — Within vs Across Participant SHAP Similarity → 03_*.png
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("ANALYSIS 3 — Participant SHAP Similarity (Overfitting Check)")
    print("=" * 60)

    # Sample up to 20 meals per participant for speed
    np.random.seed(cfg.RANDOM_SEED)
    # Positional rows only — shap_vals[i] aligns with iloc i, not arbitrary X.index labels.
    _pid_aux = pd.DataFrame({"participant_id": groups.to_numpy()})
    sampled_idx = (
        _pid_aux.groupby("participant_id", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), 20), random_state=42),
               include_groups=False)
        .index.to_numpy()
    )

    shap_sample = shap_vals[sampled_idx]
    pid_sample = participant_ids[sampled_idx]

    sim_matrix = cosine_similarity(shap_sample)

    within, across = [], []
    n_sample = len(pid_sample)
    # Use random sampling of pairs for speed if dataset is large
    max_pairs = 200_000
    if n_sample * (n_sample - 1) // 2 > max_pairs:
        rng = np.random.RandomState(42)
        for _ in range(max_pairs):
            i, j = rng.randint(0, n_sample, 2)
            if i == j:
                continue
            val = sim_matrix[i, j]
            if pid_sample[i] == pid_sample[j]:
                within.append(val)
            else:
                across.append(val)
    else:
        for i in range(n_sample):
            for j in range(i + 1, n_sample):
                val = sim_matrix[i, j]
                if pid_sample[i] == pid_sample[j]:
                    within.append(val)
                else:
                    across.append(val)

    within = np.array(within)
    across = np.array(across)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(within, bins=40, alpha=0.6,
            label=f"Within-participant (n={len(within):,})", color=COLOUR["G"])
    ax.hist(across, bins=40, alpha=0.6,
            label=f"Across-participant (n={len(across):,})", color=COLOUR["Dc"])
    ax.axvline(float(np.mean(within)), color=COLOUR["G"], lw=2, ls="--")
    ax.axvline(float(np.mean(across)), color=COLOUR["Dc"], lw=2, ls="--")
    ax.set_xlabel("Cosine similarity of SHAP vectors")
    ax.set_ylabel("Count")
    ax.set_title("SHAP Profile Similarity: Within vs Across Participants")
    ax.legend()
    plt.tight_layout()
    p = shap_plot_dir / "03_within_vs_across_shap_similarity.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")

    within_mean, within_std = np.mean(within), np.std(within)
    across_mean, across_std = np.mean(across), np.std(across)
    gap = within_mean - across_mean
    print(f"  Within-participant SHAP cosine similarity: {within_mean:.3f} +/- {within_std:.3f}")
    print(f"  Across-participant SHAP cosine similarity: {across_mean:.3f} +/- {across_std:.3f}")
    print(f"  Gap (within - across): {gap:.3f}")


    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYSIS 4 — SHAP Stability Across CV Folds (CSV only)
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("ANALYSIS 4 — SHAP Stability Across CV Folds")
    print("=" * 60)

    gkf = GroupKFold(n_splits=cfg.N_FOLDS)
    fold_top10 = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        m = xgb.XGBRegressor(**train_params)
        m.fit(X.iloc[tr], y.iloc[tr])  # type: ignore[union-attr]
        exp = shap.TreeExplainer(m)
        sv = exp.shap_values(X.iloc[te])  # type: ignore[union-attr]
        top10 = (pd.Series(np.abs(sv).mean(axis=0), index=cfg.ALL_FEATURES)
                 .nlargest(10).index.tolist())
        fold_top10.append(top10)
        print(f"  Fold {fold+1} top 5: {top10[:5]}")

    counter = Counter(feat for top in fold_top10 for feat in top)
    stable_df = pd.DataFrame(counter.most_common(20), columns=["feature", "folds_in_top10"])
    print("\n  Feature stability across folds (max=10):")
    print(stable_df.to_string(index=False))
    stable_csv = cfg.RESULTS_DIR / "shap_fold_stability.csv"
    stable_df.to_csv(stable_csv, index=False)
    print(f"  Saved: {stable_csv}")


    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYSIS 5 — SHAP Correlation Heatmap (Dc Features) → 04_*.png
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("ANALYSIS 5 — SHAP Value Correlation (Dc Features)")
    print("=" * 60)

    shap_corr = pd.DataFrame(shap_vals, columns=cfg.ALL_FEATURES).corr()
    shap_corr_dc = shap_corr.loc[dc_all, dc_all]

    fig, ax = plt.subplots(figsize=(14, 12))
    mask_tri = np.triu(np.ones_like(shap_corr_dc, dtype=bool))
    sns.heatmap(shap_corr_dc, mask=mask_tri, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, annot=False, ax=ax, linewidths=0.3)
    ax.set_title("SHAP Value Correlation — Dc Features", fontsize=12)
    plt.tight_layout()
    p = shap_plot_dir / "04_shap_correlation_dc.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


    # ═══════════════════════════════════════════════════════════════════════════
    # OUTPUT SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)

    summary = pd.DataFrame({
        "feature":       cfg.ALL_FEATURES,
        "group":         feature_groups,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        "mean_shap":     shap_vals.mean(axis=0),
        "shap_std":      shap_vals.std(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    summary_csv = cfg.RESULTS_DIR / "shap_summary_table.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"  Saved: {summary_csv}")

    print("\nTop 20 features by mean |SHAP|:")
    print(summary.head(20).to_string(index=False))


    # ═══════════════════════════════════════════════════════════════════════════
    # BIOLOGICAL INTERPRETATION CHECKLIST
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("BIOLOGICAL INTERPRETATION CHECKLIST")
    print("=" * 60)

    ranked = summary.reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)

    def get_rank(feat):
        row = ranked[ranked["feature"] == feat]
        return row["rank"].values[0] if len(row) > 0 else None  # type: ignore[union-attr]

    def get_direction(feat):
        row = ranked[ranked["feature"] == feat]
        return row["mean_shap"].values[0] if len(row) > 0 else None  # type: ignore[union-attr]

    n_features = len(cfg.ALL_FEATURES)

    checks = []

    # 1. baseline_glucose_mmol rank
    r = get_rank("baseline_glucose_mmol")
    ok = r is not None and r <= 5
    checks.append(("baseline_glucose_mmol rank",
                   f"#{r}" if r else "NOT FOUND",
                   "Expected #1 or #2",
                   "OK" if ok else "RED FLAG: not in top 5"))

    # 2. CHO direction
    d = get_direction("CHO")
    ok = d is not None and d > 0
    checks.append(("CHO SHAP direction",
                   f"{d:+.4f}" if d is not None else "N/A",
                   "Positive",
                   "OK" if ok else "RED FLAG: not positive"))

    # 3. Participant-level feature leakage check (sex is the only P_COL in the model)
    r_sex = get_rank("sex")
    sex_in_top10 = r_sex is not None and r_sex <= 10
    checks.append(("sex (participant covariate) rank",
                   f"#{r_sex}" if r_sex else "NOT IN MODEL",
                   f"Bottom 50% (rank > {n_features // 2})",
                   "OK" if not sex_in_top10 else "WARNING: participant covariate dominant"))

    # 4. Within vs across SHAP similarity gap
    checks.append(("Within vs across SHAP gap",
                   f"{gap:.3f}",
                   "< 0.05",
                   "OK" if gap < 0.05 else ("WARNING" if gap < 0.15 else "RED FLAG: memorisation")))

    # 5. Feature stability
    baseline_stability = None
    cho_stability = None
    for _, row in stable_df.iterrows():
        if row["feature"] == "baseline_glucose_mmol":
            baseline_stability = row["folds_in_top10"]
        if row["feature"] == "CHO":
            cho_stability = row["folds_in_top10"]
    if baseline_stability is None:
        baseline_stability = 0
    if cho_stability is None:
        cho_stability = 0

    checks.append(("baseline_glucose_mmol stability",
                   f"{baseline_stability}/10 folds",
                   "10/10",
                   "OK" if baseline_stability == 10 else "WARNING: not stable across folds"))

    checks.append(("CHO stability",
                   f"{cho_stability}/10 folds",
                   ">= 6/10",
                   "OK" if cho_stability >= 6 else "WARNING: CHO unstable"))

    print()
    for check_name, observed, expected, verdict in checks:
        status = "PASS" if verdict.startswith("OK") else "FAIL"
        icon = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {icon} {check_name}")
        print(f"         Observed: {observed}")
        print(f"         Expected: {expected}")
        print(f"         Verdict:  {verdict}")
        print()

    print("=" * 60)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll plots saved to: {shap_plot_dir}")
    print(f"Results saved to:   {cfg.RESULTS_DIR}")




if __name__ == "__main__":
    main()
