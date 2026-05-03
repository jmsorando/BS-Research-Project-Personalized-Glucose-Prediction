"""
research_project.config
─────────────────
Single source of truth for paths, feature lists, and model defaults.
Import this everywhere. Never hardcode paths in notebooks or scripts.
"""

from pathlib import Path
from typing import Iterable

# ── Repo root ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Data ────────────────────────────────────────────────────────────────────
FEATURE_MATRIX = REPO_ROOT / "output" / "feature_matrix.csv"
SOURCE_DIR = REPO_ROOT / "source"
OUTPUT_DATA_DIR = REPO_ROOT / "output"
CGM_DIR = SOURCE_DIR / "cgm_data"
REALIGNED_TIMES = OUTPUT_DATA_DIR / "corrected_meal_times_ALL.csv"
REALIGNED_EXTRACT = OUTPUT_DATA_DIR / "patient_extract1602_realigned.csv"
SOURCE_EXTRACT_DEFAULT = SOURCE_DIR / "patient_extract1602.csv"
PIPELINE_INPUTS_META = OUTPUT_DATA_DIR / "pipeline_inputs.json"

# ── Outputs (created at runtime if missing) ────────────────────────────────
OUTPUT_DIR = REPO_ROOT / "training_outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in [MODEL_DIR, PLOT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Target & filtering ──────────────────────────────────────────────────────
TARGET = "iAUC_mmol_min"
IAUC_STATUS = "ok"
RANDOM_SEED = 42
N_FOLDS = 10

# ── Feature groups ──────────────────────────────────────────────────────────
DC_RAW = [
    "CHO",
    "STAR",
    "TOTSUG",
    "FREE_SUGAR",
    "ADDED_SUGAR",
    "GLUC",
    "FRUCT",
    "SUCR",
    "MALT",
    "LACT",
    "GALACT",
    "OLIGO",
    "PROT",
    "FAT",
    "KCALS",
    "ALCO",
    "WATER",
    "AOACFIB",
    "ENGFIB",
    "SATFAC",
    "MONOFACc",
    "POLYFACc",
    "TOTn3PFAC",
    "TOTn6PFAC",
    "FACTRANS",
    "MG",
    "ZN",
    "MN",
    "FE",
    "SE",
    "VITD",
    "CAFF",
    "totalVeg",
    "totalFruit",
]

DC_RATIOS = [
    "starch_fraction",
    "sugar_fraction",
    "free_sugar_fraction",
    "rapid_glucose_equiv",
    "intrinsic_sugar",
    "protein_cho_ratio",
    "fat_sugar_ratio",
    "protein_sugar_ratio",
    "glycaemic_brake",
    "n6_n3_ratio",
]

G_COLS = [
    "baseline_glucose_mmol",
    "past_4h_glucose_trend",
    "past_1h_glucose_mean",
    "past_1h_glucose_sd",
    "mean_glucose_24h",
    "sd_glucose_24h",
    "mage_24h",
    "conga1_24h",
    "conga2_24h",
    "modd_24h",
    "glucose_at_t_minus_15",
    "glucose_at_t_minus_30",
]

DT_COLS = [
    "past_3h_cho",
    "past_3h_sugar",
    "past_3h_fat",
    "past_3h_prot",
    "time_since_last_meal_min",
    "hour_of_day",
]

P_COLS = [
    "sex",
]

INTERACTION_COLS = [
    "cho_x_baseline_glucose",
    "cho_x_mage",
    "cho_x_time_since_last_meal",
    "cho_x_fibre",
    "cho_x_fat",
    "cho_x_protein",
    "cho_x_hour_of_day",
    "glucose_trend_x_hour_of_day",
    "mage_x_baseline_glucose",
]

LEAKAGE_COLS = [
    "iAUC_mmol_h",
    "excursion_rise_mmol",
    "excursion_peak_mmol",
    "peak_glucose_mmol",
    "time_to_peak_min",
    "glucose_at_120min_mmol",
]

ALL_FEATURES = DC_RAW + DC_RATIOS + G_COLS + DT_COLS + P_COLS + INTERACTION_COLS

INTENTIONALLY_EXCLUDED_MODEL_COLS = [
    "n_total_meals",
    "n_days_tracked",
    "mean_daily_kcal",
    "mean_daily_cho",
    "past_3h_kcal",
    "time_since_last_sig_meal_min",
    "is_breakfast",
    "is_lunch",
    "is_dinner",
    "is_snack",
    "past_1h_glucose_range",
    "cv_glucose_24h",
    "fat_cho_ratio",
    "fibre_cho_ratio",
    "cv_x_hour_of_day",
]


def discover_latest_extract(source_dir: Path = SOURCE_DIR) -> Path:
    candidates = []
    for p in source_dir.glob("*.csv"):
        name = p.name.lower()
        if "patient_extract" in name:
            score = 0
            if "filtered" in name or "corrected" in name:
                score += 10
            if "1602" in name:
                score += 5
            candidates.append((score, p.name, p))
    if not candidates:
        return SOURCE_EXTRACT_DEFAULT
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2]


def validate_feature_contract(available_columns: Iterable[str]) -> tuple[list[str], list[str]]:
    cols = set(available_columns)
    missing_model_features = [c for c in ALL_FEATURES if c not in cols]
    excluded_present = [c for c in INTENTIONALLY_EXCLUDED_MODEL_COLS if c in cols]
    return missing_model_features, excluded_present


BASELINE_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=RANDOM_SEED,
    early_stopping_rounds=50,
)

OPTUNA_N_TRIALS = 100
OPTUNA_SEARCH_SPACE = {
    "n_estimators": ("int", 200, 1000),
    "max_depth": ("int", 3, 10),
    "learning_rate": ("float", 0.01, 0.3, {"log": True}),
    "subsample": ("float", 0.6, 1.0),
    "colsample_bytree": ("float", 0.4, 1.0),
    "min_child_weight": ("int", 1, 20),
    "reg_alpha": ("float", 1e-3, 10, {"log": True}),
    "reg_lambda": ("float", 1e-3, 10, {"log": True}),
}

