# Personalised Glucose Prediction — iAUC XGBoost Pipeline

Predict postprandial glycaemic response (PPGR) for real-world meals using XGBoost, with the target variable **iAUC** (incremental Area Under the Curve, mmol·min/L) — the glucose area above the pre-meal baseline over a 2-hour window after eating.

---

## Table of Contents

- [Background](#background)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Feature Groups](#feature-groups)
- [Training Pipeline](#training-pipeline)
- [Key Results](#key-results)
- [Data Availability](#data-availability)
- [Reference](#reference)
- [Further Reading](#further-reading)

---

## Background

Postprandial glycaemic response varies dramatically between individuals eating identical meals. This project builds a machine-learning pipeline to predict PPGR from continuous glucose monitor (CGM) data linked to self-reported food diaries (MyFood24) across 68 participants and ~2,215 meal events.

The upstream pipeline inverts the conventional approach: CGM traces are used to detect glucose excursions first, then food diary entries are matched to those excursions. This avoids reliance on self-reported meal times, which are unreliable. The nadir of each excursion (local minimum just before the glucose spike) serves as the iAUC baseline G₀, validated against Singh et al. (2025).

```mermaid
flowchart LR
    A["CGM traces"] --> B["Excursion detection\n+ nadir baseline"]
    C["Food diary\n(MyFood24)"] --> D["Meal bundling"]
    B --> E["Meal–excursion\nmatching"]
    D --> E
    E --> F["corrected_meal_times_ALL.csv"]
    F --> G["Feature engineering\n(rp-build-feature-matrix)"]
    G --> H["feature_matrix.csv\n(~2.2k rows × 104 cols)"]
    H --> I["XGBoost\n(GroupKFold CV)"]
    I --> J["CV metrics\n+ SHAP + ablation"]
```

---

## Repository Structure

```
RP Cleaning 5/
│
├── pyproject.toml                   # Package metadata, dependencies, CLI entry points (`rp-*`)
├── requirements.txt                 # Editable install: `pip install -r requirements.txt` → `-e .`
│
├── src/research_project/            # Python package (import as `research_project`)
│   ├── config.py                    # Paths, feature column lists, model defaults
│   ├── pipeline.py                  # `rp-pipeline`: chained stages (realign → … → train, optional plots/SHAP)
│   ├── data_pipeline/               # realign, build_realigned_source, feature_matrix
│   ├── training/                    # train, plots
│   └── analysis/                    # shap_analysis, shap_train_exports, shap_support
│
├── source/                          # Raw input data (do not modify)
│   ├── cgm_data/                    # 95 per-participant CGM files (Dexcom)
│   │   └── CGM_<ParticipantID>.csv
│   ├── patient_extract1602.csv      # MyFood24 food diary (16,216 rows, 168 columns)
│   └── MyFood24 ID Matched(Sheet1).csv  # Participant ↔ MyFood24 ID mapping
│
├── output/                          # Pipeline outputs (Stages 1–3)
│   ├── feature_matrix.csv           # XGBoost-ready matrix (~2.2k rows, 104 columns)
│   ├── corrected_meal_times_ALL.csv # 5,200 meal events with CGM-corrected times
│   ├── patient_extract1602_realigned.csv  # Realigned food diary
│   ├── processing_report.csv        # Per-participant match summary (105 rows)
│   ├── plots/                       # Per-participant CGM overlay plots + global summary
│   └── results/                     # Pruning analysis outputs
│
├── training_outputs/                # ML run artifacts (may be committed or regenerated locally)
│   ├── models/                      # e.g. final_model.ubj, optuna_study.pkl
│   ├── results/                     # CV metrics, SHAP CSVs, best_params.json
│   └── plots/                       # ablation, OOF scatter; `shap/` = full report from `rp-shap-report`;
│                                    # `rp-train --shap` also writes flat `shap_summary.png` + `shap_dependence_top4.png`
```

---

## Quick Start

### Prerequisites

Python 3.10+ is required. From the repository root, install the package in editable mode (dependencies come from `pyproject.toml`):

```bash
pip install -r requirements.txt
# equivalent: pip install -e .
```

This exposes console commands (`rp-pipeline`, `rp-train`, `rp-realign`, …) and allows `python -m research_project...` invocations.

**Full stack in one go:** `rp-pipeline` runs realign → build-realigned-source → feature-matrix → train; add `--plots`, `--shap`, `--tune`, `--ablation`, or `--all` for optional stages. Use `--skip-to STAGE` to resume.

### Local CLI

Ensure `output/feature_matrix.csv` is present (see [Data Availability](#data-availability)), then run:

```bash
# Baseline 10-fold CV only (~4 min)
rp-train

# Baseline + Optuna hyperparameter tuning (~20–40 min)
rp-train --tune

# Baseline + SHAP analysis
rp-train --shap

# Baseline + feature-group ablation study
rp-train --ablation

# Full pipeline: tune + SHAP + ablation (~90–120 min)
rp-train --all
```

Equivalent module form: `python -m research_project.training.train [--tune] ...`

All outputs are written to `training_outputs/`.

### Standalone SHAP Analysis

For the extended SHAP suite — figures `01`–`04` under `training_outputs/plots/shap/`, plus CSV summaries (requires a trained model in `training_outputs/models/`):

```bash
rp-shap-report
# or: python -m research_project.analysis.shap_analysis
```

### Upstream Pipeline (Data Preparation)

If you need to rebuild `feature_matrix.csv` from raw data, run the three stages in order:

```bash
rp-realign                  # Stage 1: CGM meal realignment (~2 min)
rp-build-realigned-source   # Stage 2: realigned source diary (~10 sec)
rp-build-feature-matrix     # Stage 3: feature matrix + iAUC (~3 min)
```

Module form: `python -m research_project.data_pipeline.realign`, etc.

Stage 1 auto-discovers the latest `patient_extract*.csv` in `source/` and writes
the selected filename to `output/pipeline_inputs.json`. Stage 2 reuses that same
file to keep source-to-realignment chaining consistent for refreshed extracts and
new participant drops.

---

## Feature Groups

The feature matrix contains 104 columns total. Of those, 60 are model features organised into four groups, with 6 leakage columns reserved for validation only. Hand-crafted interaction terms (`cho_x_*`, etc.) are still computed in the matrix CSV but excluded from training: XGBoost discovers these relationships via tree splits, and keeping them out avoids splitting SHAP attribution away from parent features (e.g. CHO, baseline glucose). The **temporal** group keeps three features — `past_3h_cho` (second-meal CHO carryover), `time_since_last_meal_min` (recency), and `hour_of_day` (circadian). Dropped from the model (still in the CSV): `past_3h_sugar` (subset information vs. CHO), `past_3h_fat`, `past_3h_prot`. The label was changed from “diet temporal” to “temporal” because two of the three retained features are not diet-derived.

| Group | Code | Count | Description |
|-------|------|------:|-------------|
| Diet composition | **Dc** | 34 + 10 | 34 raw nutrients (CHO, FAT, PROT, KCALS, micronutrients, etc.) + 10 derived ratios (fat_cho_ratio, fibre_cho_ratio, glycaemic_brake, etc.) |
| Glycaemic context | **G** | 12 | Pre-meal CGM state: baseline_glucose_mmol, past_4h_glucose_trend, 1h stats, 24h variability metrics (MAGE, CONGA, MODD, CV) |
| Temporal | **T** | 3 | past_3h_cho (second-meal carryover), time_since_last_meal_min (temporal modifier), hour_of_day (circadian) |
| Participant | **P** | 1 | sex |
| **Leakage** | -- | 6 | **NEVER use as features:** iAUC_mmol_h, excursion_rise_mmol, peak_glucose_mmol, etc. |

Feature lists are defined in `src/research_project/config.py` — import `research_project.config` as the single source of truth for all column names.
Some engineered columns are intentionally excluded from model training to reduce
participant-specific bias and contextual over-conditioning; see
`research_project.config.INTENTIONALLY_EXCLUDED_MODEL_COLS`.

---

## Training Pipeline

### Model Configuration

| Setting | Value |
|---------|-------|
| Algorithm | XGBoost regressor (`xgboost.XGBRegressor`) |
| Validation | 10-fold **GroupKFold** (split by `participant_id` — no data leakage across participants) |
| Metrics | MAE, RMSE, R² |
| Row filter | `iauc_status == "ok"` → ~2,150 usable rows from ~2,170 with computed iAUC |
| Encoding | `sex`: Male → 0, Female → 1 (only manual encoding; XGBoost handles NaN natively) |
| Baseline hyperparams | Defined in `research_project.config.BASELINE_PARAMS` |
| Optuna HPO | 100 trials, TPE sampler; search space in `research_project.config.OPTUNA_SEARCH_SPACE` |

### CLI Flags

| Flag | What it does | Approx. time |
|------|-------------|--------------|
| *(none)* | Baseline 10-fold CV + train final model | ~4 min |
| `--tune` | + Optuna hyperparameter optimisation (100 trials) | ~40–80 min |
| `--shap` | + SHAP summary and top-4 dependence plots | ~5 min |
| `--ablation` | + Feature-group ablation (all 15 non-empty combos of Dc, G, T, P) | ~10–20 min |
| `--all` | All of the above | ~90–120 min |
| `--data PATH` | Override the default `feature_matrix.csv` path | — |

### Pipeline Steps

1. **Load & filter** — read `feature_matrix.csv`, keep rows where `iauc_status == "ok"`, encode `sex`
2. **Baseline CV** — 10-fold GroupKFold with `research_project.config.BASELINE_PARAMS`
3. **Tuning** *(optional)* — Optuna Bayesian search, save best params to `best_params.json`
4. **Final model** — retrain on all data with the active params, save to `final_model.ubj`
5. **SHAP** *(optional)* — TreeExplainer values, beeswarm + dependence plots
6. **Ablation** *(optional)* — CV for every non-empty subset of the four feature blocks (15 rows), bar chart + presence matrix + `ablation_by_feature_group.csv`

---

## Key Results

### Ablation Study

The ablation study reveals that pre-meal glycaemic context dominates prediction, while diet composition alone carries no signal:

| Feature Set | n Features | R² (mean ± SD) |
|-------------|----------:|----------------|
| G only | 12 | ~+0.15 |
| Dc only | 44 | ~−0.004 |
| T only | 3 | ~−0.011 |
| G + Dc + T + P (full) | 60 | modest improvement over G alone |

Diet composition (Dc) and temporal (T) features perform worse than predicting the population mean when used in isolation, but contribute marginally when combined with glycaemic context. SHAP analysis at the feature level explains this finding — baseline glucose and glucose variability metrics dominate the SHAP importance ranking, while individual nutrient features have near-zero mean absolute SHAP values.

> Actual ablation scores are saved to `training_outputs/results/results.csv` after each run.

---

## Data Availability

| File | Included in repo | Notes |
|------|:---:|-------|
| `output/feature_matrix.csv` | Yes (gittracked) | ~2.2k rows × 104 columns; contains participant-level data |
| `source/` (CGM + diary) | Yes | Raw input data for the upstream pipeline |
| `training_outputs/` | Partially | Model, SHAP values, and plots are generated at runtime |

The feature matrix and source data are included in this repository. If you only need to run the ML pipeline (`rp-train`), you need `output/feature_matrix.csv`. To rebuild it from scratch, you need the full `source/` directory.

---

## Reference

> Singh R, Toumi M & Salathe M (2025). Predicting postprandial glucose response from CGM data and meal composition using machine learning. *Frontiers in Nutrition* **12**: 1539118. doi: [10.3389/fnut.2025.1539118](https://doi.org/10.3389/fnut.2025.1539118)

---

## Further Reading

- [`src/research_project/config.py`](src/research_project/config.py) — Feature lists, hyperparameters, paths, and search spaces

---

## License

This project is provided for research purposes.
