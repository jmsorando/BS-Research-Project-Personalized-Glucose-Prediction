"""Scatter of participant mean iAUC vs mean Sleep Fragmentation Index."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from research_project import config as cfg

df = pd.read_csv(cfg.FEATURE_MATRIX)
df = df[df.iauc_status == cfg.IAUC_STATUS]
df = df.dropna(subset=["sleep_fragmentation_index", cfg.TARGET])

agg = (
    df.groupby("participant_id")
    .agg(
        mean_iauc=(cfg.TARGET, "mean"),
        mean_sfi=("sleep_fragmentation_index", "mean"),
        n_meals=(cfg.TARGET, "size"),
    )
    .reset_index()
)
agg = agg[agg.n_meals >= 5]

x = agg.mean_sfi.values
y = agg.mean_iauc.values
sizes = agg.n_meals.values

pearson_r, pearson_p = stats.pearsonr(x, y)
spearman_r, spearman_p = stats.spearmanr(x, y)
slope, intercept, _, _, _ = stats.linregress(x, y)
xs = np.linspace(x.min(), x.max(), 100)
ys = slope * xs + intercept

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
    x, y,
    s=np.clip(sizes * 4, 30, 300),
    c="#4472C4", edgecolors="white", linewidths=0.6, alpha=0.78,
)
ax.plot(xs, ys, color="#C0504D", lw=1.6, ls="--",
        label=f"linear fit  (slope={slope:+.2f})")

ax.set_xlabel("Mean Sleep Fragmentation Index (per participant)")
ax.set_ylabel(f"Mean iAUC ({cfg.TARGET[6:]})")
ax.set_title(
    f"Mean iAUC vs Sleep Fragmentation Index  "
    f"(n={len(agg)} participants, ≥5 meals each)"
)

stats_txt = (
    f"Pearson  r = {pearson_r:+.3f}   (p = {pearson_p:.3g})\n"
    f"Spearman ρ = {spearman_r:+.3f}   (p = {spearman_p:.3g})"
)
ax.text(
    0.98, 0.97, stats_txt, transform=ax.transAxes,
    ha="right", va="top", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9),
)
ax.text(
    0.02, 0.02, "marker size ∝ # meals per participant",
    transform=ax.transAxes, fontsize=8, style="italic", color="#555",
)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.2)
ax.legend(loc="lower right", frameon=False, fontsize=9, bbox_to_anchor=(0.98, 0.08))
plt.tight_layout()

cfg.ensure_output_dirs()
for ext in ("png", "svg"):
    p = cfg.PLOT_DIR / f"iAUC_vs_SFI_participant.{ext}"
    plt.savefig(p, dpi=200, bbox_inches="tight")
    print(f"  Saved: {p}")
plt.close()

print()
print(f"Pearson  r = {pearson_r:+.4f}  (p = {pearson_p:.4g})")
print(f"Spearman ρ = {spearman_r:+.4f}  (p = {spearman_p:.4g})")
print(f"Linear fit:  iAUC = {slope:+.3f} * SFI + {intercept:+.2f}")
print(f"Participants in plot: {len(agg)}/{df.participant_id.nunique()}")
