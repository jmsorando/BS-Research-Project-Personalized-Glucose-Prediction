"""
make_pub_figure1.py
───────────────────
Publication Figure 1: iAUC distribution + per-participant boxplots.
Reads output/feature_matrix.csv (filtered to iauc_status == 'ok').
Saves PNG + SVG to output/plots/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
FM   = ROOT / "output" / "feature_matrix.csv"
OUT  = ROOT / "output" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

UNIT = "mmol·min/L"

df = pd.read_csv(FM)
df = df[df["iauc_status"] == "ok"].copy()
y  = df["iAUC_mmol_min"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Panel (a) — histogram
ax1.hist(y, bins=40, color="#4472C4", edgecolor="white", linewidth=0.3)
ax1.axvline(y.mean(),   color="red",    linestyle="--", linewidth=1.2, label="Mean")
ax1.axvline(y.median(), color="orange", linestyle="--", linewidth=1.2, label="Median")
ax1.set_xlabel(f"iAUC ({UNIT})")
ax1.set_ylabel("Number of meal events")
ax1.set_title("(a) iAUC distribution")
ax1.legend(frameon=False)

# Panel (b) — boxplot per participant (first 15 by sorted ID)
participants = sorted(df["participant_id"].unique())[:15]
groups = [df.loc[df["participant_id"] == p, "iAUC_mmol_min"].values for p in participants]
ax2.boxplot(groups, labels=participants, showfliers=True)
ax2.set_xlabel("Participant ID")
ax2.set_ylabel(f"iAUC ({UNIT})")
ax2.set_title("(b) iAUC per participant (sample)")
ax2.tick_params(axis="x", rotation=90)

plt.tight_layout()

for ext in ("png", "svg"):
    p = OUT / f"pub_figure1_iAUC_distribution.{ext}"
    plt.savefig(p, dpi=200, bbox_inches="tight")
    print(f"  Saved: {p}")

plt.close()
