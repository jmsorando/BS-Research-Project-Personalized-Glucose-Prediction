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


def ensure_output_dirs() -> None:
    """Create output directories on demand (call before writing files)."""
    for d in [MODEL_DIR, PLOT_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# ── Target & filtering ──────────────────────────────────────────────────────
TARGET = "iAUC_mmol_min"
IAUC_STATUS = "ok"
RANDOM_SEED = 42
N_FOLDS = 10

# ── Feature groups ──────────────────────────────────────────────────────────
DC_RAW = [
    "totalVeg",
    "totalFruit",
    "EDPOR",
    "SPECGRAV",
    "SOLD",
    "NCF",
    "GCF",
    "WATER",
    "TOTNIT",
    "PROT",
    "FAT",
    "CHO",
    "KCALS",
    "KJ",
    "STAR",
    "OLIGO",
    "TOTSUG",
    "GLUC",
    "GALACT",
    "FRUCT",
    "SUCR",
    "MALT",
    "LACT",
    "ALCO",
    "ENGFIB",
    "AOACFIB",
    "SATFAC",
    "SATFOD",
    "TOTn6PFAC",
    "TOTn6PFOD",
    "TOTn3PFAC",
    "TOTn3PFOD",
    "MONOFACc",
    "MONOFODc",
    "MONOFAC",
    "MONOFOD",
    "POLYFACc",
    "POLYFODc",
    "POLYFAC",
    "POLYFOD",
    "SATFACx6",
    "SATFODx6",
    "TOTBRFAC",
    "TOTBRFOD",
    "FACTRANS",
    "FODTRANS",
    "CHOL",
    "NA",
    "K",
    "CA",
    "MG",
    "P",
    "FE",
    "CU",
    "ZN",
    "CL",
    "MN",
    "SE",
    "I",
    "RET",
    "CAREQU",
    "RETEQU",
    "VITD",
    "VITE",
    "VITK1",
    "THIA",
    "RIBO",
    "NIAC",
    "TRYP60",
    "NIACEQU",
    "VITB6",
    "VITB12",
    "FOLT",
    "PANTO",
    "BIOT",
    "VITC",
    "ALTRET",
    "13CISRET",
    "DEHYRET",
    "RETALD",
    "ACAR",
    "BCAR",
    "CRYPT",
    "LUT",
    "LYCO",
    "25OHD3",
    "VITD3",
    "5METHF",
    "ATOPH",
    "BTOPH",
    "DTOPH",
    "GTOPH",
    "ATOTR",
    "DTOTR",
    "GTOTR",
    "FAC14:0",
    "FAC16:0",
    "FOD14:0",
    "FOD16:0",
    "FAC20:3cn6",
    "FAC20:4cn6",
    "FAC20:5cn3",
    "FAC22:6cn3",
    "FOD20:3cn6",
    "FOD20:5cn3",
    "FOD22:6cn3",
    "Total PHYTO",
    "Other CHOL and PHYTO",
    "PHYTO",
    "BSITPHYTO",
    "BRASPHYTO",
    "CAMPHYTO",
    "D5AVEN",
    "D7AVEN",
    "D7STIG",
    "STIGPHYTO",
    "CITA",
    "MALA",
    "ASH",
    "VITA_RAE",
    "VITA",
    "CARTBEQ",
    "FOLDFE",
    "FREE_SUGAR",
    "ADDED_SUGAR",
    "CAFF",
]

DC_RATIOS = []

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

T_COLS = [
    "past_3h_cho",
    "time_since_last_meal_min",
    "hour_of_day",
]

P_COLS = [
    "sex",
]

INTERACTION_COLS = []

LEAKAGE_COLS = [
    "iAUC_mmol_h",
    "excursion_rise_mmol",
    "excursion_peak_mmol",
    "peak_glucose_mmol",
    "time_to_peak_min",
    "glucose_at_120min_mmol",
]

ALL_FEATURES = DC_RAW + DC_RATIOS + G_COLS + T_COLS + P_COLS

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
    "past_3h_sugar",
    "past_3h_fat",
    "past_3h_prot",
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

