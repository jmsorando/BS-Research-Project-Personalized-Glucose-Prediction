"""
Demographic summary ("Table 1") for the ML training cohort.

Filters the participant demographics workbook to the participants that survive
the iauc_status filter in the feature matrix (the cohort the model is trained
on), decodes the quantised birth-sex codes via the patient extract, and writes
a single long-format summary CSV alongside the other result tables.

The workbook is resolved in order: ``source/Copia de ABP_participant_summary.xlsx``
(if present), else the bundled copy under ``src/research_project/data_pipeline/``
(so ``rp-demographics`` works when ``source/`` is gitignored or empty).

After `pip install -e .`:
    rp-demographics
    # or: python -m research_project.analysis.demographics
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research_project import config as cfg

# ── Inputs ──────────────────────────────────────────────────────────────────
_DEMOGRAPHICS_XLSX_NAME = "Copia de ABP_participant_summary.xlsx"
_PACKAGE_DEMOGRAPHICS_FALLBACK = (
    cfg.REPO_ROOT / "src" / "research_project" / "data_pipeline" / _DEMOGRAPHICS_XLSX_NAME
)
DEMOGRAPHICS_SHEET = "Sheet1"


def resolve_demographics_workbook() -> Path:
    """Prefer ``source/``; fall back to the copy shipped next to the data pipeline code."""
    candidates = (
        cfg.SOURCE_DIR / _DEMOGRAPHICS_XLSX_NAME,
        _PACKAGE_DEMOGRAPHICS_FALLBACK,
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Demographics workbook not found. Place it at:\n"
        f"  {candidates[0]}\n"
        "or restore the repository copy at:\n"
        f"  {candidates[1]}"
    )
MYFOOD24_BRIDGE = cfg.SOURCE_DIR / "MyFood24 ID Matched(Sheet1).csv"
PATIENT_EXTRACT = cfg.SOURCE_EXTRACT_DEFAULT  # source/patient_extract1602.csv

# Workbook columns we read.
COL_PID = "participant_id"
COL_AGE = "demo_age_response"
COL_HEIGHT = "demo_height_cm_response"
COL_WEIGHT = "demo_weight_kg_response"
COL_SEX_Q = "demo_birth_sex_quantised"
COL_ETHN = "demo_ethnicity_response"
COL_OCC = "demo_sector_response"

# Bridge / extract columns.
BRIDGE_PID = "Participant ID"
BRIDGE_MFID = "MyFood24 ID"
EXTRACT_PID = "Patient Id"
EXTRACT_SEX = "Sex"


def _load_cohort_ids() -> set[str]:
    fm = pd.read_csv(cfg.FEATURE_MATRIX, usecols=["participant_id", "iauc_status"])
    cohort = fm.loc[fm["iauc_status"] == cfg.IAUC_STATUS, "participant_id"].unique()
    return set(map(str, cohort))


def _load_demographics(cohort_ids: set[str], workbook: Path) -> pd.DataFrame:
    """Return one row per cohort participant. Missing participants become all-NaN rows."""
    df = pd.read_excel(workbook, sheet_name=DEMOGRAPHICS_SHEET)
    df[COL_PID] = df[COL_PID].astype(str)
    df = df[df[COL_PID].isin(cohort_ids)].drop_duplicates(subset=[COL_PID]).copy()

    missing = sorted(cohort_ids - set(df[COL_PID]))
    if missing:
        print(
            f"  WARNING: {len(missing)} cohort participants missing from workbook: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    # Reindex on the full cohort so absent participants become all-NaN rows; this
    # keeps N total == cohort size and lets categorical "Missing" counts reflect
    # both intra-row NaN and entirely missing participants.
    full = (
        df.set_index(COL_PID)
        .reindex(sorted(cohort_ids))
        .reset_index()
    )
    return full


def _build_sex_map() -> dict[str, str]:
    """Participant ID → 'Male'/'Female' via MyFood24 ID → Patient Id → Sex."""
    bridge = pd.read_csv(MYFOOD24_BRIDGE)
    extract = pd.read_csv(PATIENT_EXTRACT, usecols=[EXTRACT_PID, EXTRACT_SEX])
    extract = extract.drop_duplicates(subset=[EXTRACT_PID])

    bridge[BRIDGE_MFID] = pd.to_numeric(bridge[BRIDGE_MFID], errors="coerce")
    extract[EXTRACT_PID] = pd.to_numeric(extract[EXTRACT_PID], errors="coerce")

    merged = bridge.merge(
        extract,
        left_on=BRIDGE_MFID,
        right_on=EXTRACT_PID,
        how="left",
    )
    merged = merged.dropna(subset=[EXTRACT_SEX])
    merged = merged.drop_duplicates(subset=[BRIDGE_PID])
    return dict(zip(merged[BRIDGE_PID].astype(str), merged[EXTRACT_SEX].astype(str)))


def _decode_sex(demo: pd.DataFrame, sex_map: dict[str, str]) -> pd.Series:
    pid = demo[COL_PID].astype(str)
    decoded = pid.map(sex_map)
    fallback = demo[COL_SEX_Q].map({1: "1 (raw)", 2: "2 (raw)"})
    return decoded.fillna(fallback)


def _fmt_continuous(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "n/a (n=0)"
    return f"{s.mean():.1f} ± {s.std(ddof=1):.1f} (n={len(s)})"


def _category_rows(variable: str, series: pd.Series) -> list[dict[str, object]]:
    counts = series.dropna().value_counts()
    rows = [
        {"Variable": variable, "Category": str(cat), "Value": int(n)}
        for cat, n in counts.items()
    ]
    n_missing = int(series.isna().sum())
    if n_missing:
        rows.append({"Variable": variable, "Category": "Missing", "Value": n_missing})
    return rows


def _build_summary(demo: pd.DataFrame, sex_decoded: pd.Series) -> pd.DataFrame:
    age = pd.to_numeric(demo[COL_AGE], errors="coerce")
    height = pd.to_numeric(demo[COL_HEIGHT], errors="coerce")
    weight = pd.to_numeric(demo[COL_WEIGHT], errors="coerce")
    bmi = weight / (height / 100.0) ** 2

    rows: list[dict[str, object]] = [
        {"Variable": "N total", "Category": "—", "Value": len(demo)},
        {"Variable": "Age (years)", "Category": "—", "Value": _fmt_continuous(age)},
        {"Variable": "Height (cm)", "Category": "—", "Value": _fmt_continuous(height)},
        {"Variable": "Weight (kg)", "Category": "—", "Value": _fmt_continuous(weight)},
        {"Variable": "BMI (kg/m²)", "Category": "—", "Value": _fmt_continuous(bmi)},
    ]
    rows.extend(_category_rows("Sex", sex_decoded))
    rows.extend(_category_rows("Ethnicity", demo[COL_ETHN]))
    rows.extend(_category_rows("Occupation", demo[COL_OCC]))
    return pd.DataFrame(rows, columns=["Variable", "Category", "Value"])


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: rp-demographics  |  python -m research_project.analysis.demographics")
        print("  Writes a 'Table 1' summary of the ML training cohort to")
        print("  training_outputs/results/demographics_summary.csv")
        return

    cfg.ensure_output_dirs()

    print("=" * 60)
    print("DEMOGRAPHIC SUMMARY — ML TRAINING COHORT")
    print("=" * 60)

    cohort_ids = _load_cohort_ids()
    print(f"  Cohort size (iauc_status == {cfg.IAUC_STATUS!r}): {len(cohort_ids)}")

    workbook = resolve_demographics_workbook()
    print(f"  Demographics workbook: {workbook}")

    demo = _load_demographics(cohort_ids, workbook)
    matched = int(demo[COL_AGE].notna().sum())
    print(f"  Demographics rows matched: {matched} / {len(cohort_ids)}")

    sex_map = _build_sex_map()
    print(f"  Sex bridge entries: {len(sex_map)}")
    sex_decoded = _decode_sex(demo, sex_map)
    raw_fallback = sex_decoded.str.endswith("(raw)").sum() if sex_decoded.dtype == object else 0
    if raw_fallback:
        print(f"  WARNING: {raw_fallback} participants fell back to raw 1/2 codes")

    summary = _build_summary(demo, sex_decoded)

    out_path: Path = cfg.RESULTS_DIR / "demographics_summary.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n  Saved: {out_path}")

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
