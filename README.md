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

Postprandial glycaemic response varies dramatically between individuals eating identical meals. This project builds a machine-learning pipeline to predict PPGR from continuous glucose monitor (CGM) data linked to self-reported food diaries (MyFood24) across many participants and meal events.

The upstream pipeline inverts the conventional approach: CGM traces are used to detect glucose excursions first, then food diary entries are matched to those excursions. This avoids reliance on self-reported meal times, which are unreliable. The iAUC baseline is anchored at the **corrected meal time** (with a hybrid rule using the pre-excursion nadir when available); see `feature_matrix.py` and `config.py` for the exact definition.

```mermaid
flowchart LR
    A["CGM traces"] --> B["Excursion detection\n+ corrected times"]
    C["Food diary\n(MyFood24)"] --> D["Meal bundling"]
    B --> E["Meal–excursion\nmatching"]
    D --> E
    E --> F["corrected_meal_times_ALL.csv"]
    F --> G["Feature engineering\n(rp-build-feature-matrix)"]
    G --> H["feature_matrix.csv\n(~3.3k rows × ~192 cols)"]
    H --> I["XGBoost\n(GroupKFold CV)"]
    I --> J["CV metrics\n+ SHAP + ablation"]
```

---

## Repository Structure

```
 BSc Research Project/
│
├── pyproject.toml                   # Package metadata, dependencies, CLI entry points (`rp-*`)
├── requirements.txt                 # Editable install: `pip install -r requirements.txt` → `-e .`
│
├── src/research_project/            # Python package (import as `research_project`)
│   ├── config.py                    # Paths, feature column lists, model defaults
│   ├── pipeline.py                  # `rp-pipeline`: chained stages (realign → … → train, optional plots/SHAP)
│   ├── data_pipeline/               # realign, build_realigned_source, feature_matrix
│   ├── training/                    # train, plots
│   └── analysis/                    # shap_analysis, shap_train_exports, shap_support, demographics
│
├── source/                          # Raw input data (do not modify)
│   ├── cgm_data/                    # Per-participant CGM files (Dexcom)
│   │   └── CGM_<ParticipantID>.csv
│   ├── patient_extract1602.csv      # MyFood24 food diary
│   ├── MyFood24 ID Matched(Sheet1).csv  # Participant ↔ MyFood24 ID mapping
│   ├── ABP Participant Features.csv   # Overnight sleep / HRV / PSQI (merged into feature matrix)
│   └── Copia de ABP_participant_summary.xlsx  # Optional copy for `rp-demographics` (see Quick Start)
│
├── output/                          # Pipeline outputs (Stages 1–3)
│   ├── feature_matrix.csv           # XGBoost-ready matrix (~3.3k rows, ~192 columns)
│   ├── corrected_meal_times_ALL.csv # Meal events with CGM-corrected times
│   ├── patient_extract1602_realigned.csv  # Realigned food diary
│   ├── processing_report.csv        # Per-participant match summary from realign (not read by train/SHAP)
│   └── plots/                       # Optional CGM overlay plots from upstream stages
│
├── training_outputs/                # ML run artifacts (may be committed or regenerated locally)
│   ├── models/                      # e.g. final_model.ubj, optuna_study.pkl
│   ├── results/                     # CV metrics, SHAP CSVs, best_params.json
│   └── plots/                       # ablation, OOF scatter; `shap/` = full report from `rp-shap-report`;
│                                    # `rp-train --shap` also writes flat `shap_summary.png` + `shap_dependence_top4.png`
```

`rp-demographics` resolves the ABP participant-summary workbook from `source/` first, then the bundled file under `src/research_project/data_pipeline/` if needed.

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

With `--tune`, the train stage already writes the publication iAUC + OOF figures, so `rp-pipeline` skips a redundant `rp-plots` stage. With `--shap`, it runs the full `rp-shap-report` after training and does **not** pass `--shap` into `rp-train` (avoids duplicate SHAP computation); use `rp-train --shap` only when you want the lightweight flat SHAP plots without the full report.

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

### Demographics summary (Table 1 style)

Requires `output/feature_matrix.csv`. The workbook is loaded from `source/Copia de ABP_participant_summary.xlsx` when that file exists; otherwise from `src/research_project/data_pipeline/Copia de ABP_participant_summary.xlsx` (bundled fallback).

```bash
rp-demographics
# or: python -m research_project.analysis.demographics
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

The feature matrix CSV has on the order of **~190 columns** (ids, target, quality flags, validation/leakage columns, participant stratification fields, and model features). **Model features** are defined in `config.py` as `ALL_FEATURES` (currently **155** columns): diet composition (**Dc**), glycaemic context (**G**), temporal (**T**), participant (**P**), and sleep (**S**, nightly ABP metrics plus PSQI / chronotype traits). Six **leakage** columns exist for validation only and are never passed to the model.

Hand-crafted interaction columns are **not** part of the current pipeline output; XGBoost discovers interactions via tree splits. Several engineered columns remain in the CSV for diagnostics but are listed in `INTENTIONALLY_EXCLUDED_MODEL_COLS` and are not used at training time.

| Group | Code | Count | Description |
|-------|------|------:|-------------|
| Diet composition | **Dc** | 126 | Raw nutrients and related diary-derived composition (CHO, FAT, PROT, KCALS, micronutrients, etc.) |
| Glycaemic context | **G** | 12 | Pre-meal CGM state: baseline_glucose_mmol, past_4h_glucose_trend, 1h stats, 24h variability metrics (MAGE, CONGA, MODD, …) |
| Temporal | **T** | 3 | past_3h_cho, time_since_last_meal_min, hour_of_day |
| Participant | **P** | 1 | sex |
| Sleep | **S** | 13 | Nightly sleep / HRV from ABP plus psqi_scored, csm_total |
| **Leakage** | -- | 6 | **NEVER use as features:** iAUC_mmol_h, excursion_rise_mmol, peak_glucose_mmol, etc. |

Feature lists are defined in `src/research_project/config.py` — import `research_project.config` as the single source of truth for all column names.

---

## Training Pipeline

### Model Configuration

| Setting | Value |
|---------|-------|
| Algorithm | XGBoost regressor (`xgboost.XGBRegressor`) |
| Validation | 10-fold **GroupKFold** (split by `participant_id` — no data leakage across participants) |
| Metrics | MAE, RMSE, R² |
| Row filter | `iauc_status == "ok"` |
| Encoding | `sex`: Male → 0, Female → 1 (only manual encoding; XGBoost handles NaN natively) |
| Baseline hyperparams | Defined in `research_project.config.BASELINE_PARAMS` |
| Optuna HPO | 100 trials, TPE sampler; search space in `research_project.config.OPTUNA_SEARCH_SPACE` |

### CLI Flags

| Flag | What it does | Approx. time |
|------|-------------|--------------|
| *(none)* | Baseline 10-fold CV + train final model | ~4 min |
| `--tune` | + Optuna hyperparameter optimisation (100 trials) | ~40–80 min |
| `--shap` | + SHAP summary and top-4 dependence plots | ~5 min |
| `--ablation` | + Feature-group ablation (31 non-empty subsets of Dc, G, T, P, S) | ~10–20 min |
| `--all` | All of the above | ~90–120 min |
| `--data PATH` | Override the default `feature_matrix.csv` path | — |

### Pipeline Steps

1. **Load & filter** — read `feature_matrix.csv`, keep rows where `iauc_status == "ok"`, encode `sex`
2. **Baseline CV** — 10-fold GroupKFold with `research_project.config.BASELINE_PARAMS`
3. **Tuning** *(optional)* — Optuna Bayesian search, save best params to `best_params.json`
4. **Final model** — retrain on all data with the active params, save to `final_model.ubj`
5. **SHAP** *(optional)* — TreeExplainer values, beeswarm + dependence plots
6. **Ablation** *(optional)* — CV for every non-empty subset of the five feature blocks (31 rows), bar chart + presence matrix + `ablation_by_feature_group.csv`

---

## Key Results

### Ablation Study

The ablation study reveals that pre-meal glycaemic context dominates prediction, while diet composition alone carries limited signal in isolation:

| Feature Set | n Features | R² (mean ± SD) |
|-------------|----------:|----------------|
| G only | 12 | strong vs. mean |
| Dc only | 126 | weak in isolation |
| T only | 3 | weak in isolation |
| Full model (Dc + G + T + P + S) | 155 | best combined score |

> Actual ablation scores are saved to `training_outputs/results/results.csv` after each run.

---

## Data Availability

| File | Included in repo | Notes |
|------|:---:|------|
| `output/feature_matrix.csv` | Yes (gittracked) | ~3.3k rows × ~192 columns; contains participant-level data |
| `source/` (CGM + diary + ABP) | Yes | Raw input data for the upstream pipeline |
| `training_outputs/` | Partially | Model, SHAP values, and plots are generated at runtime |

The feature matrix and source data are included in this repository. If you only need to run the ML pipeline (`rp-train`), you need `output/feature_matrix.csv`. To rebuild it from scratch, you need the full `source/` directory including `ABP Participant Features.csv`.

---

## Reference

> Singh R, Toumi M & Salathe M (2025). Predicting postprandial glucose response from CGM data and meal composition using machine learning. *Frontiers in Nutrition* **12**: 1539118. doi: [10.3389/fnut.2025.1539118](https://doi.org/10.3389/fnut.2025.1539118)

---

## Further Reading

- [`src/research_project/config.py`](src/research_project/config.py) — Feature lists, hyperparameters, paths, and search spaces

---

## License

This project is provided for research purposes.
