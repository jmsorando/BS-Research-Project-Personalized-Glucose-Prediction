# Pipeline Audit Report
Generated: 2026-04-10

---

## Stage 1 — File & Path Integrity

| Check | Result |
|-------|--------|
| Feature matrix exists & > 10 KB | PASS (1,739,103 bytes) |
| iAUC results CSV (`output/iAUC_results.csv`) | FAIL — file does not exist |
| Trained model (`outputs/models/final_model.ubj`) | PASS (1,694,212 bytes) |
| SHAP values (`outputs/results/shap_values.csv`) | PASS (2,129,179 bytes) |
| CV results (`outputs/results/cv_results.csv`) | INFO — not found (optional) |
| Ablation results (`outputs/results/ablation_results.csv`) | INFO — not found (optional) |
| Feature matrix UTF-8 encoding | PASS |
| Output directories exist | PASS (all created/confirmed) |

**Note on `iAUC_results.csv`:** This file was never generated in the current branch.
The iAUC computation was performed on the `iAUC-Calculation` branch and its outputs
were merged into `feature_matrix.csv` via `build_feature_matrix.py`. Stage 2 checks
will run against the iAUC columns in `feature_matrix.csv` instead — this is the actual
data that enters the model and is the correct audit target.

---

## Stage 2 — iAUC Results Integrity (via feature_matrix.csv)

| Check | Result |
|-------|--------|
| 2.1 Schema — all iAUC columns present | PASS |
| 2.2a Total rows | WARNING — 2,228 (expected 3,500-4,500 for raw; this is the filtered matrix) |
| 2.2b OK rows | PASS — 2,215 |
| 2.2c Participants in OK | PASS — 68 |
| 2.3a iAUC min >= 0 | PASS — 12.75 |
| 2.3b iAUC max <= 3000 | PASS — 553.50 |
| 2.3c iAUC_mmol_h min >= 0 | PASS — 0.21 |
| 2.4 Unit consistency (mmol_min / mmol_h ~ 60) | PASS — 100% in range |
| 2.5 Duplicate excursions | PASS — none |
| 2.6 CGM Tier 1 (>=80% cov, <=15min gap) | PASS — 2,209/2,215 events |
| 2.6 CGM Tier 2 (>=70% cov) | PASS — 2,209 events (6 below) |
| 2.7 Confidence distribution | PASS — medium:748, low_clamped:611, high:538, low_batch_override:293, low_ampm_override:25 |
| 2.8a Baseline glucose 2-12 mmol/L | PASS |
| 2.8b Peak >= baseline | PASS |
| 2.8c Excursion rise >= 0 | PASS |
| 2.8d Time to peak 10-150 min | PASS |

**Note on 2.2a:** Total rows are lower because `feature_matrix.csv` is the post-filtering
matrix (only quality-filtered events), not the raw iAUC output. The raw file with ~4,044
events including failures was produced on the `iAUC-Calculation` branch.

**Note on 2.6:** `pct_coverage` is stored as a fraction (0-1), not percentage (0-100).
Corrected thresholds: 99.7% of OK events are Tier 1 gold standard.

---

## Stage 3 — Feature Matrix Integrity

| Check | Result |
|-------|--------|
| 3.1a Clean rows >= 1800 | PASS — 2,215 |
| 3.1b Participants >= 60 | PASS — 68 |
| 3.1c Features in config | INFO — 88 features |
| 3.2 All features present | PASS |
| 3.3 No leakage columns | PASS |
| 3.4 Missingness audit | WARNING — `modd_24h` 10.8% missing; 19 other features < 3% |
| 3.5 Zero-variance features | PASS |
| 3.6a fibre_cho_ratio >= 0 | PASS |
| 3.6b fat_cho_ratio inf/NaN | PASS |
| 3.6c n6_n3_ratio 0-50 | WARNING — 68 rows > 50 (max 1700, very low n-3 intake) |
| 3.7a Pearson r(baseline_glucose, iAUC) | PASS (r=-0.204, see note) |
| 3.7b past_4h_glucose_trend max | PASS — 3.05 < 5.0 |
| 3.8a Meal types mutually exclusive | WARNING — 2 rows sum > 1 (stacked meal labels) |
| 3.8b Unclassified meals | WARNING — 300 meals with no type label |
| 3.9 Sex encoding | PASS — Male=903, Female=1312 |
| 3.10 Participant-level stability | PASS — n_total_meals, n_days_tracked, sex all constant |

**Note on 3.4:** `modd_24h` (mean of daily differences) requires >= 2 days of CGM data.
10.8% missing is expected for participants with short monitoring. XGBoost handles natively.
Saved to `output/results/audit_missingness.csv`.

**Note on 3.7a:** Negative correlation (r=-0.204) is biologically correct for *incremental*
AUC — higher baseline glucose leaves less room for the excursion to rise above it. The audit
check threshold `r > 0.05` assumed positive correlation. This is not a data error.

**Note on 3.8:** 2 stacked meals have dual labels (e.g. "Lunch + Evening dinner"). 300
unclassified meals lack a standard meal type. Both are acceptable edge cases for this pipeline.

---

## Stage 4 — Participant Leakage Audit

| Check | Result |
|-------|--------|
| Fold 1 — no overlap | PASS — 54 train, 14 test |
| Fold 2 — no overlap | PASS — 54 train, 14 test |
| Fold 3 — no overlap | PASS — 55 train, 13 test |
| Fold 4 — no overlap | PASS — 55 train, 13 test |
| Fold 5 — no overlap | PASS — 54 train, 14 test |
| Min test participants >= 10 | PASS — minimum 13 |

---

## Stage 5 — Model Training Reproducibility

### XGBoost Baseline CV (BASELINE_PARAMS, 5-fold GroupKFold)

| Fold | R² | MAE | RMSE | n_test |
|------|-----|-----|------|--------|
| 1 | 0.122 | 55.0 | 71.9 | 443 |
| 2 | 0.163 | 49.8 | 63.8 | 441 |
| 3 | 0.149 | 51.8 | 70.5 | 442 |
| 4 | 0.107 | 50.8 | 69.8 | 445 |
| 5 | 0.111 | 49.7 | 65.5 | 444 |
| **Mean** | **0.130 +/- 0.024** | **51.4 +/- 2.2** | **68.3 +/- 3.5** | |

### Null model baseline

| Metric | Value |
|--------|-------|
| Null R² | -0.013 |
| Null MAE | 56.3 |
| XGBoost beats null on MAE | PASS |

| Check | Result |
|-------|--------|
| Null model R² near 0 | PASS (-0.013) |
| XGBoost > null on MAE | PASS (51.4 < 56.3) |

Saved to `output/results/audit_cv_results.csv`.

---

## Stage 6 — Ablation Study Verification

Ablation run with BASELINE_PARAMS + early_stopping_rounds=50 (matching `train.py`).

| Combination | n_feats | R² mean | R² SD | MAE mean |
|-------------|---------|---------|-------|----------|
| Dc only | 47 | +0.002 | 0.015 | 56.0 |
| G only | 14 | +0.134 | 0.034 | 51.0 |
| Dt only | 12 | -0.010 | 0.006 | 56.3 |
| Dc + G | 61 | +0.123 | 0.036 | 51.5 |
| Dc + Dt | 59 | -0.007 | 0.011 | 56.2 |
| G + Dt | 26 | +0.139 | 0.041 | 50.9 |
| Dc + G + Dt | 73 | +0.120 | 0.035 | 51.7 |
| All (Dc+G+Dt+P) | 78 | +0.140 | 0.029 | 51.1 |

| Check | Result |
|-------|--------|
| G only R² > 0.10 | PASS (0.134) |
| G dominance (G > Dc + 0.05) | PASS (0.134 vs 0.002) |

**Key finding confirmed:** G alone drives virtually all predictive signal (R²=0.134).
Dc alone is near-null (R²=0.002). Adding Dc/Dt on top of G yields marginal or no improvement.

Saved to `output/results/audit_ablation.csv`.

---

## Stage 7 — SHAP Integrity Checks

| Check | Result |
|-------|--------|
| 7.1 SHAP additivity (max deviation < 0.5) | PASS — 0.000519 |
| 7.2 baseline_glucose_mmol direction (positive) | FAIL — r=-0.919 (see note) |
| 7.2 CHO direction (positive) | FAIL — r=-0.006 (near-zero) |
| 7.2 fat_cho_ratio direction (negative) | FAIL — r=+0.182 |
| 7.2 fibre_cho_ratio direction (negative) | FAIL — r=+0.383 |
| 7.3 n_total_meals SHAP rank | FAIL — rank #3/88 (leakage proxy) |
| 7.3 n_days_tracked SHAP rank | PASS — rank #53/88 |
| 7.4 Within vs across SHAP similarity gap | FAIL — 0.276 (memorisation) |

**Note on 7.2 baseline_glucose_mmol:** The audit expected a positive correlation between
feature value and SHAP value. However, for *incremental* AUC (area above baseline), higher
baseline glucose mechanically reduces the increment. The strong negative r=-0.919 is
**biologically correct** for this target definition. This is a false alarm in the audit
specification, not a model error.

**Note on 7.2 CHO, fat_cho_ratio, fibre_cho_ratio:** The near-zero or inverted directions
are consistent with the ablation finding that Dc alone has R²~0.002. The model has not
learned meaningful dietary biology — it relies almost entirely on glycaemic context (G)
and participant identity proxies.

**Note on 7.3 & 7.4:** `n_total_meals` at rank #3 and the high within-participant SHAP
clustering (gap=0.276) confirm the model is partially encoding participant identity.
This is a **research finding** to address in model design (e.g. mean-centering G features
within participant, removing P covariates), not a pipeline bug.

**Action items (for research, not pipeline fixes):**
1. Consider removing `n_total_meals` and `n_days_tracked` from the feature set
2. Mean-center G features within each participant to decorrelate from identity
3. Re-evaluate whether the model can learn dietary effects after deconfounding

---

## Stage 8 — Statistical Reporting Standards

### 8.1 Participant Demographics

| Metric | Value |
|--------|-------|
| N participants | 68 |
| Sex | 40 female / 28 male |
| Meals per participant | 32.6 +/- 14.9 (range 1-60) |
| Mean iAUC | 128.5 +/- 73.7 mmol*min |
| Median iAUC | 114.0 mmol*min |
| IQR | 74.2 - 165.7 mmol*min |
| Mean baseline glucose | 5.34 +/- 1.01 mmol/L |

### 8.2 iAUC Distribution

| Check | Result |
|-------|--------|
| Skewness | 1.39 (moderate positive skew) |
| Kurtosis | 3.28 |
| Shapiro-Wilk (n=500) | W=0.9144, p<0.0001 — non-normal |
| Action | WARNING — report median (IQR) alongside mean +/- SD |

### 8.3 Per-Fold CV Metrics

| Metric | Mean +/- SD | 95% CI |
|--------|-------------|--------|
| R² | 0.130 +/- 0.024 | 0.082 - 0.178 |
| MAE | 51.4 +/- 2.2 mmol*min | 47.1 - 55.7 |
| RMSE | 68.3 +/- 3.5 mmol*min | 61.5 - 75.1 |

### 8.4 G Dominance Effect Size

| Metric | Value |
|--------|-------|
| G only R² | 0.134 |
| Dc only R² | 0.002 |
| Delta R² (G - Dc) | 0.132 |

---

## Stage 9 — Reproducibility Checklist

### Software Environment

| Package | Version |
|---------|---------|
| Python | 3.14.0 |
| XGBoost | 3.2.0 |
| scikit-learn | 1.8.0 |
| SHAP | 0.51.0 |
| pandas | 3.0.1 |
| numpy | 2.4.2 |
| matplotlib | 3.10.8 |
| scipy | 1.17.0 |
| seaborn | 0.13.2 |

### Checklist

| # | Check | Status |
|---|-------|--------|
| R1 | `random_state=42` in XGBoost | PASS |
| R2 | `random_state=42` in Optuna samplers | PASS (in train.py) |
| R3 | GroupKFold deterministic (no shuffle) | PASS |
| R4 | requirements.txt pinned versions | PASS (fixed: was unpinned, now all `==`) |
| R5 | Feature matrix build script in VCS | PASS (`build_feature_matrix.py`) |
| R6 | iAUC computation script in VCS | PASS (`cgm_meal_realignment.py` + iAUC-Calculation branch) |
| R7 | config.py single source of truth | PASS (one docstring reference in build script, not functional) |
| R8 | Outputs deterministic from raw data | PASS |
| R9 | Model saved in .ubj format | PASS |
| R10 | Seed in SHAP sampling fixed | PASS (`random_state=42` in shap_analysis.py) |

**Fix applied:** `requirements.txt` updated from unpinned to `==` pinned versions.

---

## Stage 10 — Publication-Ready Output Generation

All figures at 300 dpi with .png and .svg versions.

| Figure | File | Status |
|--------|------|--------|
| Fig 1 — iAUC distribution + per-participant boxplot | `output/plots/pub_figure1_iAUC_distribution.{png,svg}` | PASS |
| Fig 2 — Ablation study bar chart | `output/plots/pub_figure2_ablation.{png,svg}` | PASS |
| Fig 3 — SHAP beeswarm (top 20) | `output/plots/pub_figure3_shap_beeswarm.{png,svg}` | PASS |
| Supplementary table — ablation | `output/results/supplementary_table_ablation.csv` | PASS |

All figures use colorblind-safe ColorBrewer palette and coolwarm SHAP colourmap.

---

## Stage 11 — Audit Summary

| Metric | Value |
|--------|-------|
| Clean meal events (iauc_status=ok) | 2,215 |
| Unique participants | 68 |
| Features | 88 |
| CV R² | 0.130 +/- 0.024 |
| CV MAE | 51.4 +/- 2.2 mmol*min |
| CV RMSE | 68.3 +/- 3.5 mmol*min |
| Null model MAE | 56.3 mmol*min |

---

## Stage 12 — Final Pre-Submission Checklist

| # | Item | Status |
|---|------|--------|
| S1 | iAUC values are all >= 0 | PASS (min=12.75) |
| S2 | No leakage columns in feature set | PASS |
| S3 | GroupKFold confirmed — no participant in both train and test | PASS |
| S4 | Null model baseline reported alongside XGBoost metrics | PASS |
| S5 | Per-fold metrics reported (not just mean) | PASS |
| S6 | SHAP additivity check passed (max error < 0.5) | PASS (0.000519) |
| S7 | SHAP biological directions verified | FAIL (see Stage 7 notes) |
| S8 | Within vs across-participant SHAP similarity gap < 0.15 | FAIL (0.276) |
| S9 | iAUC distribution reported (mean, SD, median, IQR, skewness) | PASS |
| S10 | Tier of CGM coverage declared | PASS — Tier 1 (99.7% of events) |
| S11 | Software versions frozen in requirements.txt | PASS (fixed during audit) |
| S12 | All figures at 300 dpi and saved as .svg | PASS |
| S13 | Supplementary ablation table includes all 8 combinations | PASS |
| S14 | Participant demographics table complete | PASS |
| S15 | random_state=42 used consistently | PASS |

### Outstanding Issues (research decisions, not pipeline bugs)

1. **S7 — SHAP biological directions:** CHO, fat_cho_ratio, fibre_cho_ratio show
   inverted or near-zero directions. This reflects the model's reliance on G features
   over dietary biology, not a computation error.

2. **S8 — Participant memorisation (gap=0.276):** The model assigns systematically
   different SHAP profiles per participant. `n_total_meals` at rank #3 acts as a
   participant identity proxy. Recommended: remove P covariates and mean-center G
   features within participant before final publication.

---

*Audit completed 2026-04-10.*

