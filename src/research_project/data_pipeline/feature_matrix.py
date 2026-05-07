#!/usr/bin/env python3
"""
Build XGBoost feature matrix for iAUC prediction.

Each row is one glucose excursion and target is 2h postprandial iAUC.
"""
from collections import Counter
from datetime import timedelta
from typing import cast

import numpy as np
import pandas as pd

from research_project import config as cfg

CGM_DIR = cfg.CGM_DIR
MEAL_FILE = cfg.REALIGNED_TIMES
EXTRACT_FILE = cfg.REALIGNED_EXTRACT
OUTPUT_FILE = cfg.FEATURE_MATRIX
WINDOW_MIN = 120
EXPECTED_READS = 25
SENTINEL_LOW = 2.2
SENTINEL_HIGH = 22.2
MAX_NADIR_SNAP = 10
GAP_FLAG_MIN = 15
# Enrichment pulls the full configured Dc raw block.
NUTRIENT_COLS = list(cfg.DC_RAW)

# CSV schema: model features from cfg plus pipeline-only / validation / stratification columns.
# baseline_glucose_mmol is emitted in quality_cols (step 3), not repeated here.
G_FEATURE_MATRIX_COLS = [c for c in cfg.G_COLS if c != "baseline_glucose_mmol"] + [
    "past_1h_glucose_range",
    "cv_glucose_24h",
]
DC_RATIO_FEATURE_MATRIX_COLS = []
DT_FEATURE_MATRIX_COLS = [
    "past_3h_kcal",
    "past_3h_cho",
    "past_3h_sugar",
    "past_3h_fat",
    "past_3h_prot",
    "time_since_last_meal_min",
    "time_since_last_sig_meal_min",
    "hour_of_day",
    "is_breakfast",
    "is_lunch",
    "is_dinner",
    "is_snack",
]
INTERACTION_FEATURE_MATRIX_COLS = []


def _to_float_series(s: pd.Series) -> pd.Series:
    """Narrow pd.to_numeric for type checkers (pandas 3 stubs)."""
    return cast(pd.Series, pd.to_numeric(s, errors="coerce"))


def _scalar_float_from_numeric(val: object) -> float:
    """Single-cell numeric coercion with stable typing for pyright."""
    num = cast(pd.Series, pd.to_numeric(pd.Series([val]), errors="coerce"))
    v = num.iloc[0]
    return float(v) if pd.notna(v) else np.nan


def _parse_tz_offset(s):
    try:
        s = str(s).strip()
        sign = 1 if s[0] == "+" else -1
        pts = s[1:].split(":")
        return timedelta(hours=sign * int(pts[0]), minutes=sign * int(pts[1]) if len(pts) > 1 else 0)
    except Exception:
        return timedelta(0)


def _load_cgm(path):
    cgm = pd.read_csv(path, low_memory=False)
    if "event_type" in cgm.columns:
        cgm = cgm[cgm["event_type"] == "EGV"].copy()
    cgm["ts_utc"] = pd.to_datetime(cgm["isoDate"], utc=True, errors="coerce")
    cgm["glucose"] = cgm["glucose"].replace({"Low": SENTINEL_LOW, "High": SENTINEL_HIGH})
    cgm["glucose"] = pd.to_numeric(cgm["glucose"], errors="coerce")
    cgm = cgm.dropna(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)
    return cgm[["ts_utc", "glucose"]].copy()


def _compute_iauc(cgm_window, g0):
    ts = cgm_window["ts_utc"].values
    gl = cgm_window["glucose"].values
    iauc = 0.0
    for i in range(len(gl) - 1):
        if np.isnan(gl[i]) or np.isnan(gl[i + 1]):
            continue
        d0 = max(gl[i] - g0, 0.0)
        d1 = max(gl[i + 1] - g0, 0.0)
        dt = (ts[i + 1] - ts[i]) / np.timedelta64(1, "m")
        iauc += 0.5 * (d0 + d1) * dt
    return iauc


def _normalise_date(d):
    try:
        return pd.Timestamp(d).strftime("%Y-%m-%d")
    except Exception:
        return None


def _nearest_glucose(cgm, target_utc, max_snap_min=3):
    if cgm is None or len(cgm) == 0:
        return np.nan
    diffs = (cgm["ts_utc"] - target_utc).abs()
    idx = diffs.idxmin()
    snap = diffs.loc[idx].total_seconds() / 60
    if snap > max_snap_min:
        return np.nan
    v = cgm.loc[idx, "glucose"]
    if pd.isna(v):
        return np.nan
    return round(float(v), 4)


def _compute_mage(gl):
    if len(gl) < 8:
        return np.nan
    sd = np.std(gl, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    extrema_vals = []
    for i in range(1, len(gl) - 1):
        if gl[i] > gl[i - 1] and gl[i] > gl[i + 1]:
            extrema_vals.append(gl[i])
        elif gl[i] < gl[i - 1] and gl[i] < gl[i + 1]:
            extrema_vals.append(gl[i])
    if len(extrema_vals) < 2:
        return np.nan
    amplitudes = [abs(extrema_vals[j] - extrema_vals[j - 1]) for j in range(1, len(extrema_vals))]
    qualifying = [a for a in amplitudes if a > sd]
    if len(qualifying) == 0:
        return np.nan
    return np.mean(qualifying)


def _compute_lagged_diffs(cgm, window, lag_minutes, tol_minutes=15, min_pairs=4):
    ts_all = cgm["ts_utc"].values
    gl_all = cgm["glucose"].values
    ts_win = window["ts_utc"].values
    gl_win = window["glucose"].values
    tol_ns = np.timedelta64(int(tol_minutes), "m")
    lag_ns = np.timedelta64(int(lag_minutes), "m")
    diffs = []
    for i in range(len(ts_win)):
        if np.isnan(gl_win[i]):
            continue
        target = ts_win[i] - lag_ns
        time_diffs = np.abs(ts_all - target)
        nearest_idx = np.argmin(time_diffs)
        if time_diffs[nearest_idx] <= tol_ns and not np.isnan(gl_all[nearest_idx]):
            diffs.append(abs(float(gl_win[i]) - float(gl_all[nearest_idx])))
    if len(diffs) < min_pairs:
        return np.nan
    return np.mean(diffs)


def step1_nutrient_enrichment(meals, extract):
    print("\n  Step 1: Nutrient Enrichment")
    print("  " + "-" * 40)
    extract = extract.copy()
    extract["_date_norm"] = extract["Date"].apply(_normalise_date)
    extract["_pid"] = extract["Patient Id"]
    extract["_time_consumed"] = pd.to_datetime(extract["Time consumed at"], format="%H:%M", errors="coerce")
    nutrient_frame = pd.DataFrame(np.nan, index=meals.index, columns=NUTRIENT_COLS)
    meals = pd.concat([meals, nutrient_frame], axis=1)
    n_success = 0
    n_fail = 0
    n_mismatch = 0
    mismatches = []
    for idx in meals.index:
        pid = meals.loc[idx, "myfood24_id"]
        date_str = _normalise_date(meals.loc[idx, "date"])
        food_str = str(meals.loc[idx, "food_items"])
        if food_str == "nan" or not food_str.strip():
            n_fail += 1
            continue
        food_names = [f.strip() for f in food_str.split(";")]
        day_items = extract[(extract["_pid"] == pid) & (extract["_date_norm"] == date_str)].copy()
        if len(day_items) == 0:
            n_fail += 1
            continue
        try:
            evt_time = pd.Timestamp(meals.loc[idx, "reported_time"])
            evt_minutes = evt_time.hour * 60 + evt_time.minute
        except Exception:
            evt_minutes = None
        used_indices = set()
        matched_rows = []
        for fname in food_names:
            fname_clean = fname.strip()
            if not fname_clean:
                continue
            candidates = day_items[(day_items["Food name"] == fname_clean) & (~day_items.index.isin(used_indices))]
            if len(candidates) == 0:
                continue
            if len(candidates) == 1:
                best = candidates.index[0]
            else:
                if evt_minutes is not None and candidates["_time_consumed"].notna().any():
                    cand_minutes = candidates["_time_consumed"].dt.hour * 60 + candidates["_time_consumed"].dt.minute
                    time_diffs = (cand_minutes - evt_minutes).abs()
                    best = time_diffs.idxmin()
                else:
                    best = candidates.index[0]
            used_indices.add(best)
            matched_rows.append(best)
        if len(matched_rows) == 0:
            n_fail += 1
            continue
        for col in NUTRIENT_COLS:
            if col in extract.columns:
                vals = _to_float_series(extract.loc[matched_rows, col])
                meals.loc[idx, col] = round(float(vals.sum()), 4)
        n_success += 1
        enriched_cho = _scalar_float_from_numeric(meals.loc[idx, "CHO"])
        orig_cho = _scalar_float_from_numeric(meals.loc[idx, "total_CHO"])
        if not np.isnan(enriched_cho) and not np.isnan(orig_cho):
            if abs(enriched_cho - orig_cho) > 5:
                n_mismatch += 1
                if len(mismatches) < 5:
                    mismatches.append((meals.loc[idx, "event_id"], enriched_cho, orig_cho))
    total = n_success + n_fail
    pct = 100 * n_success / total if total > 0 else 0
    print(f"    Matched: {n_success}/{total} events ({pct:.1f}%)")
    print(f"    CHO mismatches (>5g): {n_mismatch}")
    if mismatches:
        for eid, ec, oc in mismatches:
            print(f"      {eid}: enriched={ec:.1f}g vs original={oc:.1f}g")
    return meals


def step2_aggregate_stacking(meals):
    print("\n  Step 2: Excursion-Level Aggregation")
    print("  " + "-" * 40)
    has_exc = meals[meals["excursion_id"].notna() & (meals["excursion_id"] != "")].copy()
    no_exc = meals[meals["excursion_id"].isna() | (meals["excursion_id"] == "")].copy()
    grouped_rows = []
    n_stacked_groups = 0
    for (pid, eid), grp in has_exc.groupby(["participant_id", "excursion_id"]):
        anchor = grp[grp["match_type"] == "anchor"]
        if len(anchor) == 0:
            anchor = grp.iloc[[0]]
        else:
            anchor = anchor.iloc[[0]]
        row = anchor.iloc[0].copy()
        if len(grp) > 1:
            n_stacked_groups += 1
            for col in NUTRIENT_COLS:
                if col in grp.columns:
                    vals = _to_float_series(grp[col])
                    row[col] = round(float(vals.sum()), 4)
            row["food_items"] = "; ".join(grp["food_items"].dropna().astype(str))
            row["meal_label"] = " + ".join(grp["meal_label"].dropna().astype(str))
            row["event_id"] = "; ".join(grp["event_id"].dropna().astype(str))
            row["n_items"] = float(_to_float_series(grp["n_items"]).sum())
        row["n_meals_in_excursion"] = len(grp)
        grouped_rows.append(row)
    df_exc = pd.DataFrame(grouped_rows)
    no_exc = no_exc.copy()
    no_exc["n_meals_in_excursion"] = 1
    result = pd.concat([df_exc, no_exc], ignore_index=True)
    print(f"    Excursions with stacking: {n_stacked_groups}")
    print(f"    Events with excursion_id: {len(has_exc)} -> {len(df_exc)} rows")
    print(f"    Events without excursion_id: {len(no_exc)} rows")
    print(f"    Total rows: {len(result)}")
    return result


def step3_compute_iauc(df, cgm_cache, cgm_files):
    print("\n  Step 3: iAUC Computation")
    print("  " + "-" * 40)
    iauc_out_cols = [
        "baseline_glucose_mmol",
        "iAUC_mmol_min",
        "iAUC_mmol_h",
        "peak_glucose_mmol",
        "time_to_peak_min",
        "glucose_at_120min_mmol",
        "n_readings",
        "pct_coverage",
        "max_gap_min",
        "iauc_status",
    ]
    for c in iauc_out_cols:
        if c not in df.columns:
            df[c] = np.nan
    if "iauc_status" not in df.columns or df["iauc_status"].dtype != object:
        df["iauc_status"] = ""
    status_counts = Counter()
    pids = df["participant_id"].unique()
    for pi, pid in enumerate(sorted(pids)):
        sub = df[df["participant_id"] == pid]
        if pid not in cgm_cache:
            cgm_cache[pid] = _load_cgm(cgm_files[pid]) if pid in cgm_files else None
        cgm = cgm_cache[pid]
        for idx in sub.index:
            nadir = df.loc[idx, "nadir_time"]
            if pd.isna(nadir) or str(nadir).strip() == "":
                conf = str(df.loc[idx, "confidence"])
                df.loc[idx, "iauc_status"] = conf
                status_counts[conf] += 1
                continue
            if cgm is None or len(cgm) == 0:
                df.loc[idx, "iauc_status"] = "no_cgm_file"
                status_counts["no_cgm_file"] += 1
                continue
            tz_off = _parse_tz_offset(df.loc[idx, "tz_offset"])
            nadir_utc = pd.Timestamp(nadir).tz_localize(None) - tz_off
            nadir_utc = nadir_utc.tz_localize("UTC")
            diffs = (cgm["ts_utc"] - nadir_utc).abs()
            nearest_idx = diffs.idxmin()
            snap_min = diffs.loc[nearest_idx].total_seconds() / 60
            if snap_min > MAX_NADIR_SNAP:
                df.loc[idx, "iauc_status"] = "insufficient_cgm"
                status_counts["insufficient_cgm"] += 1
                continue
            g0 = cgm.loc[nearest_idx, "glucose"]
            if np.isnan(g0):
                df.loc[idx, "iauc_status"] = "insufficient_cgm"
                status_counts["insufficient_cgm"] += 1
                continue
            window_end = nadir_utc + pd.Timedelta(minutes=WINDOW_MIN)
            mask = (cgm["ts_utc"] >= nadir_utc - pd.Timedelta(seconds=30)) & (
                cgm["ts_utc"] <= window_end + pd.Timedelta(seconds=30)
            )
            window = cgm.loc[mask].copy()
            if len(window) < 3:
                df.loc[idx, "iauc_status"] = "insufficient_cgm"
                status_counts["insufficient_cgm"] += 1
                continue
            n_readings = len(window)
            pct_cov = n_readings / EXPECTED_READS
            gaps = window["ts_utc"].diff().dt.total_seconds() / 60
            max_gap = gaps.max() if len(gaps) > 1 else 0.0
            iauc = _compute_iauc(window, g0)
            valid_gl = window.dropna(subset=["glucose"])
            if len(valid_gl) > 0:
                peak_idx = valid_gl["glucose"].idxmax()
                peak_gl = valid_gl.loc[peak_idx, "glucose"]
                peak_ts = valid_gl.loc[peak_idx, "ts_utc"]
                ttp = (peak_ts - nadir_utc).total_seconds() / 60
            else:
                peak_gl = np.nan
                ttp = np.nan
            end_diffs = (window["ts_utc"] - window_end).abs()
            end_idx = end_diffs.idxmin()
            end_snap = end_diffs.loc[end_idx].total_seconds() / 60
            gl_120 = window.loc[end_idx, "glucose"] if end_snap <= 10 else np.nan
            status = "gap_too_large" if max_gap > GAP_FLAG_MIN else "ok"
            df.loc[idx, "baseline_glucose_mmol"] = round(g0, 4)
            df.loc[idx, "iAUC_mmol_min"] = round(iauc, 4)
            df.loc[idx, "iAUC_mmol_h"] = round(iauc / 60, 4)
            df.loc[idx, "peak_glucose_mmol"] = round(peak_gl, 4)
            df.loc[idx, "time_to_peak_min"] = round(ttp, 1)
            df.loc[idx, "glucose_at_120min_mmol"] = round(gl_120, 4) if not np.isnan(gl_120) else np.nan
            df.loc[idx, "n_readings"] = int(n_readings)
            df.loc[idx, "pct_coverage"] = round(pct_cov, 4)
            df.loc[idx, "max_gap_min"] = round(max_gap, 1)
            df.loc[idx, "iauc_status"] = status
            status_counts[status] += 1
        if (pi + 1) % 10 == 0 or pi + 1 == len(pids):
            print(f"    [{pi+1}/{len(pids)}] {pid}")
    computed = status_counts.get("ok", 0) + status_counts.get("gap_too_large", 0)
    print(f"    Computed iAUC for {computed} rows")
    for s, n in status_counts.most_common():
        print(f"      {s:30s} {n:>5d}")
    return df, cgm_cache


def step4_glycaemic_features(df, cgm_cache):
    print("\n  Step 4: Glycaemic Features (G)")
    print("  " + "-" * 40)
    g_cols = [
        "past_4h_glucose_trend",
        "past_1h_glucose_mean",
        "past_1h_glucose_sd",
        "past_1h_glucose_range",
        "mean_glucose_24h",
        "sd_glucose_24h",
        "cv_glucose_24h",
        "mage_24h",
        "conga1_24h",
        "conga2_24h",
        "modd_24h",
        "glucose_at_t_minus_15",
        "glucose_at_t_minus_30",
    ]
    for c in g_cols:
        df[c] = np.nan
    n_computed = 0
    for pid in sorted(df["participant_id"].unique()):
        cgm = cgm_cache.get(pid)
        if cgm is None or len(cgm) == 0:
            continue
        for idx in df[df["participant_id"] == pid].index:
            nadir = df.loc[idx, "nadir_time"]
            if pd.isna(nadir) or str(nadir).strip() == "":
                continue
            tz_off = _parse_tz_offset(df.loc[idx, "tz_offset"])
            t0_utc = pd.Timestamp(nadir).tz_localize(None) - tz_off
            t0_utc = t0_utc.tz_localize("UTC")
            mask_4h = (cgm["ts_utc"] >= t0_utc - pd.Timedelta(hours=4)) & (cgm["ts_utc"] <= t0_utc)
            window_4h = cgm.loc[mask_4h].dropna(subset=["glucose"])
            if len(window_4h) >= 6:
                x = (window_4h["ts_utc"] - t0_utc).dt.total_seconds().values / 3600
                y = window_4h["glucose"].values
                x_mean, y_mean = x.mean(), y.mean()
                var_x = ((x - x_mean) ** 2).sum()
                if var_x > 0:
                    slope = ((x - x_mean) * (y - y_mean)).sum() / var_x
                    df.loc[idx, "past_4h_glucose_trend"] = round(slope, 4)
            mask_1h = (cgm["ts_utc"] >= t0_utc - pd.Timedelta(hours=1)) & (cgm["ts_utc"] <= t0_utc)
            window_1h = cgm.loc[mask_1h].dropna(subset=["glucose"])
            if len(window_1h) >= 2:
                gl = window_1h["glucose"]
                df.loc[idx, "past_1h_glucose_mean"] = round(gl.mean(), 4)
                df.loc[idx, "past_1h_glucose_sd"] = round(gl.std(), 4)
                df.loc[idx, "past_1h_glucose_range"] = round(gl.max() - gl.min(), 4)
            mask_24h = (cgm["ts_utc"] >= t0_utc - pd.Timedelta(hours=24)) & (cgm["ts_utc"] <= t0_utc)
            window_24h = cgm.loc[mask_24h].dropna(subset=["glucose"])
            if len(window_24h) >= 12:
                gl24 = window_24h["glucose"]
                m24, s24 = gl24.mean(), gl24.std()
                df.loc[idx, "mean_glucose_24h"] = round(m24, 4)
                df.loc[idx, "sd_glucose_24h"] = round(s24, 4)
                df.loc[idx, "cv_glucose_24h"] = round(s24 / m24, 4) if m24 > 0 else np.nan
            if len(window_24h) >= 8:
                mage = _compute_mage(window_24h["glucose"].values)
                if not np.isnan(mage):
                    df.loc[idx, "mage_24h"] = round(mage, 4)
            conga1 = _compute_lagged_diffs(cgm, window_24h, lag_minutes=60)
            if not np.isnan(conga1):
                df.loc[idx, "conga1_24h"] = round(conga1, 4)
            conga2 = _compute_lagged_diffs(cgm, window_24h, lag_minutes=120)
            if not np.isnan(conga2):
                df.loc[idx, "conga2_24h"] = round(conga2, 4)
            modd = _compute_lagged_diffs(cgm, window_24h, lag_minutes=1440)
            if not np.isnan(modd):
                df.loc[idx, "modd_24h"] = round(modd, 4)
            df.loc[idx, "glucose_at_t_minus_15"] = _nearest_glucose(cgm, t0_utc - pd.Timedelta(minutes=15), 3)
            df.loc[idx, "glucose_at_t_minus_30"] = _nearest_glucose(cgm, t0_utc - pd.Timedelta(minutes=30), 3)
            n_computed += 1
    print(f"    Computed glycaemic features for {n_computed} rows")
    for c in g_cols:
        pct = 100 * df[c].notna().sum() / len(df) if len(df) > 0 else 0
        print(f"      {c:30s} {pct:5.1f}% non-null")
    return df


def step5_diet_temporal(df, all_meals):
    print("\n  Step 5: Temporal Features (T)")
    print("  " + "-" * 40)
    dt_cols = [
        "past_3h_kcal",
        "past_3h_cho",
        "past_3h_sugar",
        "past_3h_fat",
        "past_3h_prot",
        "time_since_last_meal_min",
        "time_since_last_sig_meal_min",
        "hour_of_day",
        "is_breakfast",
        "is_lunch",
        "is_dinner",
        "is_snack",
    ]
    for c in dt_cols:
        df[c] = np.nan
    all_meals = all_meals.copy()
    all_meals["_ct"] = pd.to_datetime(all_meals["corrected_time"], errors="coerce")
    all_meals["_cho_val"] = pd.to_numeric(all_meals["total_CHO"], errors="coerce")
    all_meals["_kcal_val"] = pd.to_numeric(all_meals["total_KCALS"], errors="coerce")
    all_meals["_totsug_val"] = pd.to_numeric(all_meals["total_TOTSUG"], errors="coerce")
    all_meals["_fat_val"] = pd.to_numeric(all_meals["total_FAT"], errors="coerce")
    all_meals["_prot_val"] = pd.to_numeric(all_meals["total_PROT"], errors="coerce")
    meals_by_pid = {pid: grp.sort_values("_ct").reset_index(drop=True) for pid, grp in all_meals.groupby("participant_id")}
    n_computed = 0
    for idx in df.index:
        pid = df.loc[idx, "participant_id"]
        try:
            t0 = pd.Timestamp(df.loc[idx, "corrected_time"])
        except Exception:
            continue
        df.loc[idx, "hour_of_day"] = t0.hour
        ml = str(df.loc[idx, "meal_label"]).lower()
        df.loc[idx, "is_breakfast"] = 1 if "breakfast" in ml else 0
        df.loc[idx, "is_lunch"] = 1 if "lunch" in ml else 0
        df.loc[idx, "is_dinner"] = 1 if "evening dinner" in ml or "dinner" in ml else 0
        has_main = "breakfast" in ml or "lunch" in ml or "dinner" in ml
        df.loc[idx, "is_snack"] = 1 if ("snack" in ml and not has_main) else 0
        p_meals = meals_by_pid.get(pid)
        if p_meals is None or len(p_meals) == 0:
            n_computed += 1
            continue
        prior = p_meals[p_meals["_ct"] < t0]
        cutoff_3h = t0 - pd.Timedelta(hours=3)
        recent = prior[prior["_ct"] >= cutoff_3h]
        if len(recent) > 0:
            df.loc[idx, "past_3h_kcal"] = round(recent["_kcal_val"].sum(), 2)
            df.loc[idx, "past_3h_cho"] = round(recent["_cho_val"].sum(), 2)
            df.loc[idx, "past_3h_sugar"] = round(recent["_totsug_val"].sum(), 2)
            df.loc[idx, "past_3h_fat"] = round(recent["_fat_val"].sum(), 2)
            df.loc[idx, "past_3h_prot"] = round(recent["_prot_val"].sum(), 2)
        else:
            df.loc[idx, ["past_3h_kcal", "past_3h_cho", "past_3h_sugar", "past_3h_fat", "past_3h_prot"]] = 0
        if len(prior) > 0:
            df.loc[idx, "time_since_last_meal_min"] = round((t0 - prior["_ct"].iloc[-1]).total_seconds() / 60, 1)
        sig_prior = prior[prior["_cho_val"] > 10]
        if len(sig_prior) > 0:
            df.loc[idx, "time_since_last_sig_meal_min"] = round((t0 - sig_prior["_ct"].iloc[-1]).total_seconds() / 60, 1)
        n_computed += 1
    print(f"    Computed temporal features for {n_computed} rows")
    return df


def step6_derived_ratios(df):
    print("\n  Step 6: Derived Nutrient Ratios (Dc enrichment)")
    print("  " + "-" * 40)
    cho = _to_float_series(df["CHO"]).fillna(0)
    totsug = _to_float_series(df["TOTSUG"]).fillna(0)
    star = _to_float_series(df["STAR"]).fillna(0)
    free_sug = _to_float_series(df["FREE_SUGAR"]).fillna(0)
    gluc = _to_float_series(df["GLUC"]).fillna(0)
    sucr = _to_float_series(df["SUCR"]).fillna(0)
    malt = _to_float_series(df["MALT"]).fillna(0)
    fat = _to_float_series(df["FAT"]).fillna(0)
    prot = _to_float_series(df["PROT"]).fillna(0)
    fibre = _to_float_series(df["AOACFIB"]).fillna(0)
    n6 = _to_float_series(df["TOTn6PFAC"]).fillna(0)
    n3 = _to_float_series(df["TOTn3PFAC"]).fillna(0)
    cho_safe = cho.clip(lower=0.1)
    totsug_safe = totsug.clip(lower=0.1)
    cho_safe_1 = cho.clip(lower=1.0)
    n3_safe = n3.clip(lower=0.01)
    df["starch_fraction"] = (star / cho_safe).round(4)
    df["sugar_fraction"] = (totsug / cho_safe).round(4)
    df["free_sugar_fraction"] = (free_sug / cho_safe).round(4)
    df["rapid_glucose_equiv"] = (gluc + sucr * 0.5 + malt).round(4)
    df["intrinsic_sugar"] = (totsug - free_sug).round(4)
    df["fat_cho_ratio"] = (fat / cho_safe).round(4)
    df["protein_cho_ratio"] = (prot / cho_safe).round(4)
    df["fibre_cho_ratio"] = (fibre / cho_safe).round(4)
    df["fat_sugar_ratio"] = (fat / totsug_safe).round(4)
    df["protein_sugar_ratio"] = (prot / totsug_safe).round(4)
    df["glycaemic_brake"] = ((prot * 3.27 + fat * 1.54 + fibre * 2.0) / cho_safe_1).round(4)
    df["n6_n3_ratio"] = (n6 / n3_safe).round(4)
    return df


def step7_participant_features(df, all_meals, sex_map):
    print("\n  Step 7: Participant-Level Features")
    print("  " + "-" * 40)
    all_meals = all_meals.copy()
    all_meals["_date_norm"] = all_meals["date"].apply(_normalise_date)
    all_meals["_kcal"] = pd.to_numeric(all_meals["total_KCALS"], errors="coerce")
    all_meals["_cho"] = pd.to_numeric(all_meals["total_CHO"], errors="coerce")
    plevel = []
    for pid, grp in all_meals.groupby("participant_id"):
        dates = grp["_date_norm"].dropna().unique()
        daily_kcal = grp.groupby("_date_norm")["_kcal"].sum()
        daily_cho = grp.groupby("_date_norm")["_cho"].sum()
        mfid = grp["myfood24_id"].iloc[0]
        plevel.append(
            {
                "participant_id": pid,
                "sex": sex_map.get(mfid, np.nan),
                "n_total_meals": len(grp),
                "n_days_tracked": len(dates),
                "mean_daily_kcal": round(daily_kcal.mean(), 1) if len(daily_kcal) > 0 else np.nan,
                "mean_daily_cho": round(daily_cho.mean(), 1) if len(daily_cho) > 0 else np.nan,
            }
        )
    plevel_df = pd.DataFrame(plevel)
    df = df.merge(
        plevel_df[["participant_id", "sex", "n_total_meals", "n_days_tracked", "mean_daily_kcal", "mean_daily_cho"]],
        on="participant_id",
        how="left",
        suffixes=("", "_plevel"),
    )
    if "sex_plevel" in df.columns:
        df["sex"] = df["sex_plevel"].fillna(df.get("sex", np.nan))
        df.drop(columns=["sex_plevel"], inplace=True)
    for c in ["n_total_meals", "n_days_tracked", "mean_daily_kcal", "mean_daily_cho"]:
        if f"{c}_plevel" in df.columns:
            df[c] = df[f"{c}_plevel"]
            df.drop(columns=[f"{c}_plevel"], inplace=True)
    return df


def compute_interaction_features(df):
    df["cho_x_baseline_glucose"] = df["CHO"] * df["baseline_glucose_mmol"]
    df["cho_x_mage"] = df["CHO"] * df["mage_24h"]
    df["cho_x_time_since_last_meal"] = df["CHO"] * df["time_since_last_meal_min"]
    df["cho_x_fibre"] = df["CHO"] * df["ENGFIB"]
    df["cho_x_fat"] = df["CHO"] * df["FAT"]
    df["cho_x_protein"] = df["CHO"] * df["PROT"]
    df["cho_x_hour_of_day"] = df["CHO"] * df["hour_of_day"]
    df["glucose_trend_x_hour_of_day"] = df["past_4h_glucose_trend"] * df["hour_of_day"]
    df["mage_x_baseline_glucose"] = df["mage_24h"] * df["baseline_glucose_mmol"]
    df["cv_x_hour_of_day"] = df["cv_glucose_24h"] * df["hour_of_day"]
    return df


def main():
    print("=" * 72)
    print("  Feature Matrix Builder - XGBoost iAUC Prediction")
    print("=" * 72)
    print("\n  Loading data...")
    meals = pd.read_csv(MEAL_FILE, low_memory=False)
    print(f"    Meal events: {len(meals)} rows")
    extract = pd.read_csv(EXTRACT_FILE, low_memory=False)
    print(f"    Patient extract: {len(extract)} rows")
    source_extract = cfg.discover_latest_extract(cfg.SOURCE_DIR)
    source = pd.read_csv(source_extract, usecols=["Patient Id", "Sex"], low_memory=False)
    sex_map = source.drop_duplicates("Patient Id").set_index("Patient Id")["Sex"].to_dict()
    cgm_files = {f.stem.replace("CGM_", ""): f for f in CGM_DIR.glob("CGM_*.csv")}
    print(f"    CGM files: {len(cgm_files)}")
    cgm_cache = {}
    all_meals = meals.copy()
    meals = step1_nutrient_enrichment(meals, extract)
    df = step2_aggregate_stacking(meals)
    df, cgm_cache = step3_compute_iauc(df, cgm_cache, cgm_files)
    df = step4_glycaemic_features(df, cgm_cache)
    df = step5_diet_temporal(df, all_meals)
    df = step7_participant_features(df, all_meals, sex_map)
    print("\n  Assembling output...")
    if "event_id" in df.columns:
        df.rename(columns={"event_id": "event_ids"}, inplace=True)
    if "food_items" in df.columns:
        df.rename(columns={"food_items": "food_items_concat"}, inplace=True)
    if "meal_label" in df.columns:
        df.rename(columns={"meal_label": "meal_labels"}, inplace=True)
    if "n_items" in df.columns:
        df.rename(columns={"n_items": "n_food_items"}, inplace=True)
    id_cols = ["participant_id", "excursion_id", "event_ids", "date", "meal_labels", "n_meals_in_excursion", "n_food_items", "food_items_concat"]
    target_cols = ["iAUC_mmol_min"]
    quality_cols = ["confidence", "match_type", "batch_day", "baseline_glucose_mmol", "n_readings", "pct_coverage", "max_gap_min", "iauc_status"]
    dc_nutrient_cols = NUTRIENT_COLS
    dc_ratio_cols = DC_RATIO_FEATURE_MATRIX_COLS
    g_cols = G_FEATURE_MATRIX_COLS
    dt_cols = DT_FEATURE_MATRIX_COLS
    interaction_cols = INTERACTION_FEATURE_MATRIX_COLS
    participant_cols = ["sex", "n_total_meals", "n_days_tracked", "mean_daily_kcal", "mean_daily_cho"]
    validation_cols = ["excursion_rise_mmol", "excursion_peak_mmol", "peak_glucose_mmol", "time_to_peak_min", "glucose_at_120min_mmol", "iAUC_mmol_h"]
    all_ordered = []
    seen = set()
    for col in id_cols + target_cols + quality_cols + dc_nutrient_cols + dc_ratio_cols + g_cols + dt_cols + interaction_cols + participant_cols + validation_cols:
        if col not in seen and col in df.columns:
            all_ordered.append(col)
            seen.add(col)
    df_out = df[all_ordered].copy()
    missing_model_features, excluded_present = cfg.validate_feature_contract(df_out.columns)
    if missing_model_features:
        raise ValueError(f"Missing configured model features in output schema: {missing_model_features}")
    if excluded_present:
        print("\n  Guardrail: intentionally excluded columns present (not used by train.py):")
        print(f"    {', '.join(excluded_present)}")
    n_before = len(df_out)
    df_out = df_out[df_out["iAUC_mmol_min"].notna()].reset_index(drop=True)
    print(f"\n  Filtered: {n_before} -> {len(df_out)} rows (kept only computed iAUC)")
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved: {OUTPUT_FILE.name}")
    print(f"  {len(df_out)} rows, {len(df_out.columns)} columns")


if __name__ == "__main__":
    main()

