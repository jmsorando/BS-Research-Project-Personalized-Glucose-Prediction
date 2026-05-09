"""SHAP correlation heatmap for the top 25 features overall (all groups)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from research_project import config as cfg

shap_vals = pd.read_csv(cfg.RESULTS_DIR / "shap_values.csv")
summary = pd.read_csv(cfg.RESULTS_DIR / "shap_summary_table.csv")

top = summary.head(25).copy()
top_features = top["feature"].tolist()
group_lookup = dict(zip(top["feature"], top["group"]))

shap_corr = shap_vals[top_features].corr()

group_colors = {
    "G": "#1f77b4",
    "Dc": "#2ca02c",
    "T": "#ff7f0e",
    "P": "#9467bd",
    "S": "#d62728",
    "St": "#8c564b",
}

fig, ax = plt.subplots(figsize=(13, 11))
mask_tri = np.triu(np.ones_like(shap_corr, dtype=bool))
sns.heatmap(
    shap_corr, mask=mask_tri, cmap="RdBu_r", center=0,
    vmin=-1, vmax=1, annot=False, ax=ax, linewidths=0.3,
)
ax.set_title("SHAP Value Correlation — Top 25 Features Overall (by mean |SHAP|)", fontsize=12)

for tick, label in zip(ax.get_yticklabels(), top_features):
    tick.set_color(group_colors.get(group_lookup[label], "#333"))
for tick, label in zip(ax.get_xticklabels(), top_features):
    tick.set_color(group_colors.get(group_lookup[label], "#333"))

handles = [
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=10, label=g)
    for g, c in group_colors.items()
    if g in top["group"].unique()
]
ax.legend(handles=handles, title="Feature group", loc="lower left",
          bbox_to_anchor=(1.18, 0.02), frameon=False, fontsize=9, title_fontsize=10)

plt.tight_layout()
out = cfg.PLOT_DIR / "shap" / "05_shap_correlation_top25_overall.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
print()
print("Top 25 features (group / mean |SHAP|):")
print(top[["feature", "group", "mean_abs_shap"]].to_string(index=False))
