"""
shap_analysis.py
────────────────
Comprehensive SHAP analysis for the iAUC XGBoost model.
Produces 10 analyses with plots and summary tables.

Run from repo root:
    python shap_analysis.py
"""

import sys, io, unittest.mock
from pathlib import Path

# Fix cp1252 encoding on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Block numba before shap loads it — its DLL is blocked by Windows App Control
for _mod in ["numba", "numba.core", "numba.core.decorators",
             "numba.stencils", "numba.stencils.stencil",
             "numba.core.ir_utils", "numba.core.extending",
             "numba.core.pythonapi", "numba.typed"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = unittest.mock.MagicMock()

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

# ── Output directories ───────────────────────────────────────────────────
SHAP_PLOT_DIR = cfg.PLOT_DIR / "shap"
SHAP_PLOT_DIR.mkdir(parents=True, exist_ok=True)
cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette (ColorBrewer, colorblind-safe, print-safe) ────────────
COLOUR = {
    "G":  "#2166AC",   # blue — glycaemic context
    "Dc": "#D6604D",   # red-orange — diet composition
    "Dt": "#4DAC26",   # green — diet temporal
    "P":  "#7B2D8B",   # purple — participant covariates
    "I":  "#DDCC77",   # Tol muted gold — interactions
}

# SHAP beeswarm/summary colourmap: coolwarm gives softer journal-friendly poles
SHAP_CMAP = plt.cm.coolwarm  # type: ignore[attr-defined]


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(cfg.FEATURE_MATRIX)
df = df[df["iauc_status"] == cfg.IAUC_STATUS].reset_index(drop=True)
df["sex"] = df["sex"].map({"Male": 0, "Female": 1})  # type: ignore[arg-type]

X = df[cfg.ALL_FEATURES]
y = df[cfg.TARGET]
participant_ids = df["participant_id"].values  # type: ignore[union-attr]

print(f"  Rows: {len(df):,}  Features: {X.shape[1]}  "
      f"Participants: {df['participant_id'].nunique()}")  # type: ignore[union-attr]


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
import json
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

# Load or compute SHAP values
if shap_csv.exists():
    shap_df = pd.read_csv(shap_csv)
    shap_vals = shap_df.values
    print(f"  Loaded SHAP values from {shap_csv}")
else:
    print("  Computing SHAP values ...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    shap_df = pd.DataFrame(shap_vals, columns=cfg.ALL_FEATURES)
    shap_df.to_csv(shap_csv, index=False)
    print(f"  Saved SHAP values to {shap_csv}")

explainer = shap.TreeExplainer(model)
expected_value = explainer.expected_value
if hasattr(expected_value, "__len__"):
    expected_value = float(expected_value[0]) if len(expected_value) == 1 else float(np.mean(expected_value))
expected_value = float(expected_value)

preds = model.predict(X)
residuals = y.values - preds  # type: ignore[union-attr]

print(f"  SHAP matrix shape: {shap_vals.shape}")
print(f"  Expected value (baseline): {expected_value:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: group assignment for each feature
# ═══════════════════════════════════════════════════════════════════════════

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

feature_groups = [get_feature_group(f) for f in cfg.ALL_FEATURES]


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — Beeswarm Summary (Top 20)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ANALYSIS 1 — Beeswarm Summary (Top 20)")
print("=" * 60)

plt.figure(figsize=(10, 10))
shap.summary_plot(shap_vals, X, max_display=20, show=False, cmap=SHAP_CMAP)
plt.gca().set_xlabel("SHAP value (mmol·min/L)")
plt.title("SHAP Global Feature Importance", fontsize=12)
plt.tight_layout()
p = SHAP_PLOT_DIR / "01_beeswarm_top20.png"
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")
print("  Caption: Beeswarm plot showing top 20 features ranked by mean |SHAP|.")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Group-Level Bar Chart
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ANALYSIS 2 — Group-Level SHAP Bar Chart")
print("=" * 60)

mean_abs_shap = np.abs(shap_vals).mean(axis=0)
feat_importance = pd.Series(mean_abs_shap, index=cfg.ALL_FEATURES)

group_importance = {
    "G (glycaemic)":      feat_importance[cfg.G_COLS].sum(),
    "Dc (diet compo)":    feat_importance[cfg.DC_RAW + cfg.DC_RATIOS].sum(),
    "Dt (temporal)":      feat_importance[cfg.DT_COLS].sum(),
    "P (participant)":    feat_importance[cfg.P_COLS].sum(),
    "I (interactions)":   feat_importance[cfg.INTERACTION_COLS].sum(),
}

gi = pd.Series(group_importance).sort_values()
colours = [COLOUR.get(k.split(" ")[0], "#888") for k in gi.index]

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(gi.index, gi.values, color=colours)
ax.set_xlabel("Sum of mean |SHAP| (mmol·min/L)")
ax.set_title("Feature Group Importance — Sum of Mean |SHAP|")
for i, (name, val) in enumerate(gi.items()):
    ax.text(val + gi.max() * 0.01, i, f"{val:.2f}", va="center", fontsize=9)
plt.tight_layout()
p = SHAP_PLOT_DIR / "02_group_shap_bar.png"
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")
for name, val in sorted(group_importance.items(), key=lambda x: -x[1]):
    print(f"    {name}: {val:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Dependence Plots for Top 8 Features
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ANALYSIS 3 — Dependence Plots (Top 8)")
print("=" * 60)

top8 = feat_importance.nlargest(8).index.tolist()

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, (ax, feat) in enumerate(zip(axes.flat, top8)):
    shap.dependence_plot(feat, shap_vals, X, interaction_index="auto",
                         ax=ax, show=False, cmap=SHAP_CMAP)
    ax.set_ylabel("SHAP value (mmol·min/L)")
    ax.set_title(f"#{i+1} {feat}", fontsize=10)
plt.suptitle("SHAP Dependence Plots — Top 8 Features", fontsize=13, y=1.01)
plt.tight_layout()
p = SHAP_PLOT_DIR / "03_dependence_top8.png"
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")
print(f"  Top 8: {top8}")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 4 — Waterfall Plots for Representative Meals
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ANALYSIS 4 — Waterfall Plots")
print("=" * 60)

# Select 4 representative meals
high_iauc_well = None
threshold_high = y.mean() + 1.5 * y.std()
threshold_low = y.mean() - 1.0 * y.std()

# High iAUC, well predicted: small |residual| among high-iAUC meals
high_mask = y > threshold_high
if high_mask.any():
    high_indices = y[high_mask].index
    abs_res_high = np.abs(residuals[high_indices])
    high_iauc_well = high_indices[np.argmin(abs_res_high)]
else:
    high_iauc_well = y.idxmax()

# Low iAUC, well predicted: small |residual| among low-iAUC meals
low_mask = y < threshold_low
if low_mask.any():
    low_indices = y[low_mask].index
    abs_res_low = np.abs(residuals[low_indices])
    low_iauc_well = low_indices[np.argmin(abs_res_low)]
else:
    low_iauc_well = y.idxmin()

# High iAUC, badly predicted: largest positive residual
high_iauc_bad = np.argmax(residuals)

# Average meal: closest to mean
avg_meal = (y - y.mean()).abs().idxmin()

meal_cases = {
    "high_iAUC_well_predicted": high_iauc_well,
    "low_iAUC_well_predicted":  low_iauc_well,
    "high_iAUC_badly_predicted": high_iauc_bad,
    "average_meal":             avg_meal,
}

for label, idx in meal_cases.items():
    expl = shap.Explanation(
        values=shap_vals[idx],
        base_values=expected_value,
        data=X.iloc[idx].values,  # type: ignore[union-attr]
        feature_names=list(cfg.ALL_FEATURES),
    )
    fig = plt.figure(figsize=(10, 8))
    shap.waterfall_plot(expl, max_display=15, show=False)
    plt.gca().set_xlabel("SHAP value (mmol·min/L)")
    plt.title(f"Waterfall — {label}\n"
              f"actual={y.iloc[idx]:.1f}, pred={preds[idx]:.1f}", fontsize=11)  # type: ignore[union-attr]
    plt.tight_layout()
    fname = f"04_waterfall_{label}.png"
    p = SHAP_PLOT_DIR / fname
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}  (idx={idx}, actual={y.iloc[idx]:.1f}, pred={preds[idx]:.1f})")  # type: ignore[union-attr]


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 5 — CHO × Baseline Glucose Interaction
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ANALYSIS 5 — CHO × Baseline Glucose Interaction")
print("=" * 60)

# CHO coloured by baseline_glucose
if "CHO" in cfg.ALL_FEATURES and "baseline_glucose_mmol" in cfg.ALL_FEATURES:
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot(
        "CHO", shap_vals, X,
        interaction_index="baseline_glucose_mmol",
        ax=ax, show=False, cmap=SHAP_CMAP,
    )
    ax.set_ylabel("SHAP value (mmol·min/L)")
    ax.set_title("CHO SHAP value coloured by baseline glucose\n"
                 "(warm = high baseline → amplified carb response?)", fontsize=11)
    plt.tight_layout()
    p = SHAP_PLOT_DIR / "05_interaction_CHO_baseline.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")

    # Mirror: baseline_glucose coloured by CHO
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot(
        "baseline_glucose_mmol", shap_vals, X,
        interaction_index="CHO",
        ax=ax, show=False, cmap=SHAP_CMAP,
    )
    ax.set_ylabel("SHAP value (mmol·min/L)")
    ax.set_title("baseline_glucose_mmol SHAP value coloured by CHO\n"
                 "(warm = high CHO → amplified glucose effect?)", fontsize=11)
    plt.tight_layout()
    p = SHAP_PLOT_DIR / "05_interaction_baseline_CHO.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")
else:
    print("  WARNING: CHO or baseline_glucose_mmol not in ALL_FEATURES — skipping")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 6 — Within vs Across Participant SHAP Similarity
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ANALYSIS 6 — Participant SHAP Similarity (Overfitting Check)")
print("=" * 60)

# Sample up to 20 meals per participant for speed
np.random.seed(cfg.RANDOM_SEED)
sampled_idx = (df.groupby("participant_id", group_keys=False)
                 .apply(lambda g: g.sample(min(len(g), 20), random_state=42))
                 .index.to_numpy())

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
p = SHAP_PLOT_DIR / "06_within_vs_across_shap_similarity.png"
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
# ANALYSIS 7 — Dc-Only Model SHAP
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ANALYSIS 7 — Dc-Only Model SHAP")
print("=" * 60)

dc_feats = cfg.DC_RAW + cfg.DC_RATIOS
model_dc = xgb.XGBRegressor(**train_params)
model_dc.fit(X[dc_feats], y)

explainer_dc = shap.TreeExplainer(model_dc)
shap_vals_dc = explainer_dc.shap_values(X[dc_feats])

fig, ax = plt.subplots(1, 1, figsize=(9, 12))

plt.sca(ax)
shap.summary_plot(shap_vals_dc, X[dc_feats], max_display=25, show=False, cmap=SHAP_CMAP)
ax.set_xlabel("SHAP value (mmol·min/L)")
ax.set_title("SHAP — Dc-only model", fontsize=11)

plt.tight_layout()
p = SHAP_PLOT_DIR / "07_dc_only_shap.png"
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 8 — SHAP Stability Across CV Folds
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ANALYSIS 8 — SHAP Stability Across CV Folds")
print("=" * 60)

gkf = GroupKFold(n_splits=cfg.N_FOLDS)
fold_top10 = []

for fold, (tr, te) in enumerate(gkf.split(X, y, df["participant_id"])):
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
# ANALYSIS 9 — Dc SHAP by Meal Type
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ANALYSIS 9 — Dc SHAP Importance by Meal Type")
print("=" * 60)

# NOTE: is_breakfast / is_lunch / is_dinner / is_snack were removed from DT_COLS
# as model features but are retained here as stratification masks because they
# still exist in the raw feature_matrix.csv.
meal_types = {
    "Breakfast": df["is_breakfast"] == 1,
    "Lunch":     df["is_lunch"] == 1,
    "Dinner":    df["is_dinner"] == 1,
    "Snack":     df["is_snack"] == 1,
}

dc_all = cfg.DC_RAW + cfg.DC_RATIOS
dc_feats_idx = [cfg.ALL_FEATURES.index(f) for f in dc_all if f in cfg.ALL_FEATURES]

fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
for ax, (label, mask) in zip(axes, meal_types.items()):
    sv_sub = shap_vals[mask.values]  # type: ignore[union-attr]
    dc_importance = np.abs(sv_sub[:, dc_feats_idx]).mean(axis=0)
    dc_names = [cfg.ALL_FEATURES[i] for i in dc_feats_idx]
    top_idx = np.argsort(dc_importance)[-10:]
    ax.barh([dc_names[i] for i in top_idx], dc_importance[top_idx], color=COLOUR["Dc"])
    ax.set_title(f"{label}\n(n={mask.sum()})")
    ax.set_xlabel("Mean |SHAP| (mmol·min/L)")
axes[0].set_ylabel("Dc features")
plt.suptitle("Diet Composition SHAP Importance by Meal Type", fontsize=13)
plt.tight_layout()
p = SHAP_PLOT_DIR / "09_dc_shap_by_meal_type.png"
plt.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 10 — SHAP Correlation Heatmap (Dc Features)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ANALYSIS 10 — SHAP Value Correlation (Dc Features)")
print("=" * 60)

shap_corr = pd.DataFrame(shap_vals, columns=cfg.ALL_FEATURES).corr()
shap_corr_dc = shap_corr.loc[dc_all, dc_all]

fig, ax = plt.subplots(figsize=(14, 12))
mask_tri = np.triu(np.ones_like(shap_corr_dc, dtype=bool))
sns.heatmap(shap_corr_dc, mask=mask_tri, cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, annot=False, ax=ax, linewidths=0.3)
ax.set_title("SHAP Value Correlation — Dc Features", fontsize=12)
plt.tight_layout()
p = SHAP_PLOT_DIR / "10_shap_correlation_dc.png"
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
bottom_20_threshold = int(n_features * 0.8)

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

# 3. Participant leakage check
r_meals = get_rank("n_total_meals")
r_days = get_rank("n_days_tracked")
leakage_flag = False
if r_meals and r_meals <= 10:
    leakage_flag = True
if r_days and r_days <= 10:
    leakage_flag = True
checks.append(("n_total_meals / n_days_tracked rank",
               f"n_total_meals=#{r_meals}, n_days_tracked=#{r_days}",
               f"Bottom 20% (rank > {bottom_20_threshold})",
               "OK" if not leakage_flag else "RED FLAG: participant leakage suspected"))

# 5. Within vs across SHAP similarity gap
checks.append(("Within vs across SHAP gap",
               f"{gap:.3f}",
               "< 0.05",
               "OK" if gap < 0.05 else ("WARNING" if gap < 0.15 else "RED FLAG: memorisation")))

# 6. Feature stability
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
print(f"\nAll plots saved to: {SHAP_PLOT_DIR}")
print(f"Results saved to:   {cfg.RESULTS_DIR}")
